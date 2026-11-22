import speech_recognition as sr
import pandas as pd
import google.generativeai as genai
import re
import os
import time
from gtts import gTTS
import tempfile
import pygame
import threading
from datetime import datetime
import logging
from pathlib import Path
import RPi.GPIO as GPIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basmala_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPIO Configuration
GPIO.setmode(GPIO.BCM)
led_pin2 = 24
led_pin = 21
servo_pin = 13

# Ultrasonic Sensors Configuration
sensors = [
    {'name': 'Center Sensor', 'TRIG': 4, 'ECHO': 17, 'angle': 110},
    {'name': 'Left Sensor', 'TRIG': 5, 'ECHO': 27, 'angle': 180},
    {'name': 'Right Sensor', 'TRIG': 18, 'ECHO': 25, 'angle': 0}
]

# Setup GPIO pins
def setup_gpio():
    try:
        # LED
        GPIO.setup(led_pin, GPIO.OUT)
        GPIO.output(led_pin, False)
        GPIO.setup(led_pin2, GPIO.OUT)
        GPIO.output(led_pin2, False)
        
        # Servo
        GPIO.setup(servo_pin, GPIO.OUT)
        pwm = GPIO.PWM(servo_pin, 50)  # 50Hz for standard servo
        pwm.start(0)
        
        # Ultrasonic sensors
        for sensor in sensors:
            GPIO.setup(sensor['TRIG'], GPIO.OUT)
            GPIO.setup(sensor['ECHO'], GPIO.IN)
            GPIO.output(sensor['TRIG'], False)
            time.sleep(0.1)
            
        return pwm
    except Exception as e:
        logger.error(f"GPIO setup error: {e}")
        return None

# Function to get distance from ultrasonic sensor
def get_distance(sensor):
    try:
        GPIO.output(sensor['TRIG'], True)
        time.sleep(0.00001)
        GPIO.output(sensor['TRIG'], False)

        pulse_start = time.time()
        timeout = pulse_start + 0.04  # 40ms timeout (max distance ~6.8m)

        while GPIO.input(sensor['ECHO']) == 0:
            if time.time() > timeout:
                return float('inf')
            pulse_start = time.time()

        pulse_end = time.time()
        while GPIO.input(sensor['ECHO']) == 1:
            if time.time() > timeout:
                return float('inf')
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2  # Speed of sound (34300 cm/s)
        return distance
    except Exception as e:
        logger.error(f"Distance measurement error: {e}")
        return float('inf')

# Function to move servo smoothly
def smooth_move_servo(pwm, target_angle):
    try:
        target_angle = max(0, min(180, target_angle))
        current_angle = getattr(smooth_move_servo, 'last_angle', 90)

        step = 2 if target_angle > current_angle else -2
        for angle in range(current_angle, target_angle + step, step):
            duty = 2 + (angle / 18)
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.005)

        pwm.ChangeDutyCycle(0)

        setattr(smooth_move_servo, 'last_angle', target_angle)
    except Exception as e:
        logger.error(f"Servo movement error: {e}")
        pwm.ChangeDutyCycle(0)

# Check sensors for activation
def check_sensors(pwm, activation_distance=50):
    try:
        for sensor in sensors:
            dist = get_distance(sensor)
            logger.info(f"{sensor['name']} distance: {dist:.1f} cm")
            
            if dist < activation_distance:
                GPIO.output(led_pin, True)
                GPIO.output(led_pin2, False)
                
                logger.info(f"Object detected by {sensor['name']} at {dist:.1f} cm")
                smooth_move_servo(pwm, sensor['angle'])
                return True
                
        GPIO.output(led_pin, False)
        GPIO.output(led_pin2, True)
        return False  
    except Exception as e:
        logger.error(f"Sensor check error: {e}")
        return False

# Configuration for Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Global state variables
conversation_history = []
selected_experiment_no = None
current_context = {"experiment": None, "section": None, "last_action": None}
mixer_initialized = False
df = None
experiment_lookup = {}

SYSTEM_PROMPT = """
You are Basmala, a friendly and enthusiastic science experiment assistant.
Your personality traits:
- Encouraging and positive
- Explains complex concepts in simple terms
- Occasionally uses science puns or fun facts
- Shows excitement about scientific discovery
- Speaks in a warm, conversational tone
- Keeps responses concise but informative (1-2 short sentences)
- Remembers conversation context and builds on previous exchanges
- Asks follow-up questions to keep users engaged

Focus on providing accurate information based on the experiments in the Excel file.
Always stay in character as Basmala and maintain enthusiasm for science learning.
"""

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="models/gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT
)

def initialize_mixer():
    """Initialize pygame mixer with better error handling"""
    global mixer_initialized
    try:
        if not mixer_initialized:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            mixer_initialized = True
            logger.info("Audio mixer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Mixer initialization error: {e}")
        mixer_initialized = False
        return False

def load_experiment_data():
    """Load and process experiment data with better error handling"""
    global df, experiment_lookup
    
    excel_files = [
        "C1.xlsx",
    ]
    
    # Try to find and load Excel file
    for excel_path in excel_files:
        if Path(excel_path).exists():
            try:
                df = pd.read_excel(excel_path)
                logger.info(f"Successfully loaded {len(df)} experiments from {excel_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {excel_path}: {e}")
                continue
    
    if df is None:
        logger.warning("No Excel file found, creating sample data")
        create_sample_data()
        return
    
    # Process the data
    try:
        process_experiment_data()
        logger.info(f"Successfully processed {len(experiment_lookup)} experiments")
    except Exception as e:
        logger.error(f"Error processing experiment data: {e}")
        create_sample_data()

def create_sample_data():
    """Create sample experiment data for testing"""
    global df, experiment_lookup
    
    sample_data = {
        'No': [1, 2, 3, 4, 5, 6, 7, 8],
        'Experiment Name': [
            'Rainbow in a Glass',
            'Volcano Eruption', 
            'Invisible Ink',
            'Dancing Raisins',
            'Color Changing Milk',
            'Floating Egg',
            'Crystal Garden',
            'Magnetic Slime'
        ],
        'Discovery': [
            'Different liquids have different densities and create colorful layers',
            'Chemical reactions can produce gas and create explosive effects',
            'Heat can reveal hidden messages written with lemon juice',
            'Carbon dioxide bubbles make objects dance in liquid',
            'Soap molecules disrupt surface tension creating swirling colors',
            'Salt water increases density making objects float',
            'Crystals grow in patterns when conditions are right',
            'Iron filings make slime respond to magnetic fields'
        ],
        'Explanation': [
            'Density differences prevent mixing of liquids with different weights',
            'Baking soda and vinegar create an acid-base reaction producing CO2',
            'Citric acid in lemon juice becomes visible when heated',
            'CO2 bubbles attach to raisins changing their buoyancy',
            'Detergent molecules have hydrophobic and hydrophilic ends',
            'Salt increases water density beyond that of the egg',
            'Supersaturated solutions allow crystal formation on nucleation sites',
            'Iron particles align with magnetic field lines'
        ],
        'Action': [
            'Layer different colored liquids by density in a clear container',
            'Mix baking soda and vinegar in a bottle shaped like a volcano',
            'Write with lemon juice and heat the paper to reveal message',
            'Drop raisins in clear carbonated soda and watch them dance',
            'Add food coloring to milk then touch with cotton swab dipped in soap',
            'Place egg in water, then gradually add salt until it floats',
            'Suspend string in supersaturated salt solution overnight',
            'Mix iron filings into slime and use magnets to control it'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create experiment lookup
    experiment_lookup = {}
    for _, row in df.iterrows():
        exp_no = int(row['No'])
        exp_name = str(row['Experiment Name'])
        experiment_lookup[exp_no] = {
            'name': exp_name,
            'keywords': exp_name.lower().split() + exp_name.lower().replace(' ', '').split()
        }
    
    logger.info("Sample experiment data created successfully")

def process_experiment_data():
    """Process loaded experiment data"""
    global experiment_lookup
    
    # Find the correct column names
    name_column = find_column(['Experiment Name', 'Name', 'Title', 'Experiment'])
    no_column = find_column(['No', 'Number', '#', 'ID', 'Num'])
    
    if name_column is None:
        name_column = df.columns[0]
        logger.warning(f"Using first column '{name_column}' for experiment names")
    
    if no_column is None:
        df['No'] = range(1, len(df) + 1)
        no_column = 'No'
        logger.warning("Created sequential numbers for experiments")
    
    # Build experiment lookup
    experiment_lookup = {}
    for _, row in df.iterrows():
        try:
            exp_no = int(row[no_column]) if pd.notna(row[no_column]) else len(experiment_lookup) + 1
            exp_name = str(row[name_column]) if pd.notna(row[name_column]) else f"Experiment {exp_no}"
            
            experiment_lookup[exp_no] = {
                'name': exp_name,
                'keywords': exp_name.lower().split() + [exp_name.lower().replace(' ', '')]
            }
        except Exception as e:
            logger.warning(f"Error processing experiment row: {e}")
            continue

def find_column(possible_names):
    """Find column by possible names (case-insensitive)"""
    for col in df.columns:
        col_str = str(col).strip()
        for possible in possible_names:
            if possible.lower() in col_str.lower():
                return col
    return None

def clean_text_for_speech(text):
    """Clean and optimize text for speech synthesis"""
    # Remove markdown and special characters
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Fix common abbreviations and units
    replacements = {
        '°C': ' degrees Celsius',
        '°F': ' degrees Fahrenheit', 
        'ml': ' milliliters',
        'cm': ' centimeters',
        'mm': ' millimeters',
        'kg': ' kilograms',
        'g': ' grams',
        'pH': ' P H',
        'CO2': ' carbon dioxide',
        'H2O': ' water'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Limit length for better speech flow
    sentences = text.split('.')
    if len(sentences) > 2:
        text = '. '.join(sentences[:2]) + '.'
    
    return text.strip()

def speak_with_tts(text):
    """Enhanced TTS with better audio management"""
    global mixer_initialized
    
    try:
        if not mixer_initialized and not initialize_mixer():
            print(f"[AUDIO ERROR] {text}")
            return
        
        # Clean text for speech
        clean_text = clean_text_for_speech(text)
        
        # Stop any current audio
        try:
            pygame.mixer.music.stop()
            time.sleep(0.1)
        except:
            pass
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.save(temp_filename)
        
        # Play audio
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for completion
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
        
        time.sleep(0.1)  # Brief pause
        
        # Cleanup
        try:
            os.unlink(temp_filename)
        except:
            pass
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        print(f"[SPEECH ERROR] {text}")

def speak(text):
    """Main speak function with visual feedback"""
    print(f"\n?? Basmala: {text}")
    speak_with_tts(text)

def enhanced_listen(timeout=10, phrase_limit=15):
    """Enhanced speech recognition with better error handling"""
    recognizer = sr.Recognizer()
    
    # Optimize recognizer settings
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    
    try:
        with sr.Microphone() as source:
            # Stop any playing audio
            try:
                pygame.mixer.music.stop()
                time.sleep(0.2)
            except:
                pass
            
            print("?? Listening...")
            
            # Calibrate for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen for speech
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            
            print("?? Processing...")
            
            # Recognize speech
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"?? You said: {text}")
            
            return text.strip()
            
    except sr.WaitTimeoutError:
        print("? No speech detected")
        return ""
    except sr.UnknownValueError:
        print("? Could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        print("?? Speech recognition service unavailable")
        return ""
    except Exception as e:
        logger.error(f"Unexpected listening error: {e}")
        print("?? Audio processing error")
        return ""

def analyze_intent(user_input):
    """Improved intent analysis with better context awareness"""
    try:
        context_info = ""
        if current_context["experiment"]:
            exp_name = experiment_lookup.get(current_context["experiment"], {}).get('name', 'Unknown')
            context_info = f"Currently discussing experiment {current_context['experiment']}: {exp_name}. "
        
        prompt = f"""
        {context_info}
        
        Analyze this user input and classify intent: '{user_input}'
        
        Categories:
        - greeting: hello, hi, hey, start
        - experiment_request: asking for specific experiment by number or name
        - section_request: asking about discovery, explanation, action, principle
        - general_question: science questions not about specific experiments
        - clarification: tell me more, explain, what does that mean
        - navigation: list experiments, next, previous, go back
        - positive_feedback: good, great, cool, awesome, interesting
        - negative_feedback: boring, don't understand, confusing
        - farewell: bye, goodbye, exit, quit, done
        
        Return format: intent_category|extracted_info
        
        Examples:
        "Tell me about experiment 3" ? experiment_request|3
        "What's the discovery?" ? section_request|discovery
        "How does photosynthesis work?" ? general_question|photosynthesis
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip().split("|")
        
        intent = result[0].strip().lower()
        info = result[1].strip() if len(result) > 1 else ""
        
        logger.info(f"Intent: {intent}, Info: {info}")
        return intent, info
        
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        return classify_intent_fallback(user_input)

def classify_intent_fallback(user_input):
    """Fallback intent classification using keyword matching"""
    user_lower = user_input.lower()
    
    # Simple keyword-based classification
    if any(word in user_lower for word in ["hi", "hello", "hey", "start"]):
        return "greeting", ""
    elif any(word in user_lower for word in ["bye", "goodbye", "exit", "quit", "done"]):
        return "farewell", ""
    elif re.search(r'experiment\s*(\d+)', user_lower):
        return "experiment_request", re.search(r'experiment\s*(\d+)', user_lower).group(1)
    elif re.search(r'\b(\d+)\b', user_lower):
        return "experiment_request", re.search(r'\b(\d+)\b', user_lower).group(1)
    elif any(section in user_lower for section in ["discovery", "explanation", "action", "principle"]):
        for section in ["discovery", "explanation", "action", "principle"]:
            if section in user_lower:
                return "section_request", section
    elif any(word in user_lower for word in ["list", "show", "what", "available"]):
        return "navigation", "list"
    else:
        return "general_question", user_input

def find_experiments_by_query(query):
    """Find experiments matching a query"""

    query_lower = query.lower()
    matches = []
    
    for exp_no, exp_data in experiment_lookup.items():
        # Check name match
        if query_lower in exp_data['name'].lower():
            matches.append((exp_no, exp_data['name'], 'name'))
        # Check keyword match
        elif any(keyword in query_lower or query_lower in keyword for keyword in exp_data['keywords']):
            matches.append((exp_no, exp_data['name'], 'keyword'))
    
    return matches

def get_experiment_section(exp_no, section):
    """Get specific section of an experiment"""
    section_map = {
        "discovery": ["Discovery", "What happens", "Result", "Observation"],
        "explanation": ["Explanation", "Why", "Reason", "Science", "Theory"],
        "action": ["Action", "Steps", "Method", "Procedure", "How to"],
        "principle": ["Principle", "Scientific principle", "Law", "Concept"]
    }
    
    if exp_no not in experiment_lookup:
        return f"Experiment {exp_no} not found."
    
    # Find matching row
    exp_row = df[df['No'] == exp_no]
    if exp_row.empty:
        return f"No data found for experiment {exp_no}."
    
    # Find matching column
    section_columns = section_map.get(section.lower(), [section])
    content = None
    
    for col_name in section_columns:
        if col_name in df.columns:
            content = exp_row.iloc[0][col_name]
            break
        # Try partial match
        for col in df.columns:
            if col_name.lower() in str(col).lower():
                content = exp_row.iloc[0][col]
                break
        if content is not None:
            break
    
    if content is None or pd.isna(content):
        return f"The {section} section is not available for this experiment."
    
    # Update context
    current_context["section"] = section
    current_context["last_action"] = f"viewed_{section}"
    
    return str(content)

def generate_contextual_response(user_input):
    """Generate contextual response using conversation history"""
    global conversation_history
    
    # Add to history
    conversation_history.append(f"User: {user_input}")
    
    # Keep recent history
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    
    # Build context
    context_info = ""
    if current_context["experiment"]:
        exp_name = experiment_lookup.get(current_context["experiment"], {}).get('name', 'Unknown')
        context_info = f"Currently discussing experiment {current_context['experiment']}: {exp_name}. "
    
    recent_history = "\n".join(conversation_history[-6:])
    
    prompt = f"""
    Context: {context_info}
    Recent conversation: {recent_history}
    Current input: {user_input}
    
    Respond as Basmala with 1-2 enthusiastic sentences.
    Be encouraging and maintain scientific curiosity.
    If unsure about experiment details, ask for clarification.
    """
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        conversation_history.append(f"Basmala: {result}")
        return result
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return "I'm having trouble processing that right now. Could you ask me about a specific experiment?"

def handle_experiment_request(info):
    """Handle experiment selection requests"""
    global selected_experiment_no
    
    try:

        # Try to parse as number first
        exp_no = int(info)
        if exp_no in experiment_lookup:
            selected_experiment_no = exp_no
            current_context["experiment"] = exp_no
            current_context["section"] = None
            
            exp_name = experiment_lookup[exp_no]['name']
            return f"Excellent choice! Let's explore experiment {exp_no}: {exp_name}. What would you like to know - the discovery, explanation, action steps, or scientific principle?"
        else:
            available = list(experiment_lookup.keys())
            return f"I don't have experiment {exp_no}. Available experiments are numbered {min(available)} to {max(available)}."
            
    except ValueError:
        # Search by name/keywords
        matches = find_experiments_by_query(info)
        if matches:
            if len(matches) == 1:
                exp_no, exp_name, _ = matches[0]
                selected_experiment_no = exp_no
                current_context["experiment"] = exp_no
                return f"Found it! Let's explore experiment {exp_no}: {exp_name}. What aspect interests you most?"
            else:
                match_list = [f"{no}: {name}" for no, name, _ in matches[:3]]
                return f"I found several matches: {', '.join(match_list)}. Which one would you like to explore?"
        else:
            return f"I couldn't find an experiment matching '{info}'. Try asking for a specific number or describing what you're interested in!"

def list_available_experiments():
    """Create a friendly list of available experiments"""
    if not experiment_lookup:
        return "I don't have any experiments loaded right now. Please check the data file."
    
    total = len(experiment_lookup)
    sample_size = min(5, total)
    
    samples = list(experiment_lookup.items())[:sample_size]
    sample_text = ", ".join([f"{no}: {data['name']}" for no, data in samples])
    
    if total > sample_size:
        sample_text += f" and {total - sample_size} more!"
    
    return f"I have {total} fantastic experiments ready! Here are some examples: {sample_text}. Just tell me a number or describe what interests you!"

def reset_conversation():
    """Reset conversation state"""
    global conversation_history, selected_experiment_no, current_context
    conversation_history = []
    selected_experiment_no = None
    current_context = {"experiment": None, "section": None, "last_action": None}
    logger.info("Conversation state reset")

def main_conversation_loop():
    """Main conversation loop with sensor integration"""
    print("\n" + "="*60)
    print("?? BASMALA SCIENCE ASSISTANT WITH SENSOR INTEGRATION")
    print("="*60)
    
    # Initialize systems
    if not initialize_mixer():
        print("?? Audio system not available - text-only mode")
    
    load_experiment_data()
    
    # Initialize GPIO
    pwm = setup_gpio()
    if pwm is None:
        print("? Failed to initialize GPIO - exiting")
        return
    
    # Start from center
    smooth_move_servo(pwm, 90)
    
    # Welcome message
    welcome_msg = "Hello! I'm Basmala, your enthusiastic science companion! I'll activate when you approach within 50cm."
    speak(welcome_msg)
    
    conversation_active = True
    silence_count = 0
    max_silence = 3
    
    while conversation_active:
        try:
            # First check sensors - only proceed if object detected
            if not check_sensors(pwm):
                time.sleep(0.5)  # Short delay between sensor checks
                continue
                
            # Object detected - start interaction
            print("\n" + "-"*40)
            print("?? User detected - starting interaction")
            
            # Listen for user input
            user_input = enhanced_listen()
            
            if not user_input:
                silence_count += 1
                if silence_count >= max_silence:
                    speak("I'm still here whenever you're ready to explore science together!")
                    silence_count = 0
                continue
            
            silence_count = 0
            
            # Analyze intent
            intent, info = analyze_intent(user_input)
            logger.info(f"Intent: {intent}, Info: {info}")
            
            # Handle different intents
            if intent == "farewell":
                speak("Thank you for this wonderful scientific journey! Keep exploring and stay curious. See you next time!")
                conversation_active = False
                
            elif intent == "greeting":
                response = generate_contextual_response(user_input)
                speak(response)
                
            elif intent == "experiment_request":
                response = handle_experiment_request(info)
                speak(response)
                
            elif intent == "section_request":
                if current_context["experiment"]:
                    section_info = get_experiment_section(current_context["experiment"], info)
                    speak(section_info)
                else:
                    speak("Let's pick an experiment first! " + list_available_experiments())
                    
            elif intent == "navigation":
                if "list" in info:
                    speak(list_available_experiments())
                else:
                    speak("What would you like to explore? I can list experiments or help you find something specific!")
                    
            elif intent == "positive_feedback":
                speak("I'm so glad you're enjoying our scientific exploration! What else would you like to discover?")
                
            elif intent == "negative_feedback":
                speak("Let me try to explain it differently! What specific part would you like me to clarify?")
                
            elif intent == "clarification":
                if current_context["last_action"]:
                    speak("I'd be happy to explain more! What specific part would you like me to dive deeper into?")
                else:
                    speak("What would you like me to clarify? I'm here to make science clear and exciting!")
                    
            else:  # general_question
                response = generate_contextual_response(user_input)
                speak(response)
            
            # Brief pause between interactions
            time.sleep(0.3)
            
        except KeyboardInterrupt:
            print("\n\n?? Goodbye! Thanks for exploring science with me!")
            break
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            speak("I encountered a small hiccup, but I'm still here to help with your science questions!")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main_conversation_loop()
    except KeyboardInterrupt:
        print("\n\n?? Program interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"?? Critical error: {e}")
    finally:
        # Cleanup
        try:
            if mixer_initialized:
                pygame.mixer.quit()
            if 'pwm' in globals():
                pwm.ChangeDutyCycle(0) 
                time.sleep(0.5)
                pwm.stop()
            GPIO.cleanup()
        except:
            pass
        print("\n?? Basmala Assistant closed cleanly")