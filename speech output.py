import cv2
import pyttsx3
import speech_recognition as sr
import pickle
import dlib
import json
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import face_recognition
import webbrowser
import subprocess
import requests
import dateparser
from datetime import datetime
from bs4 import BeautifulSoup


app = Flask(__name__)
DATABASE_FILE = 'database.json'


engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)
voices = engine.getProperty("voices")


indian_female_voice = None
for voice in voices:
    if "Indian" in voice.name and "Female" in voice.name:
        indian_female_voice = voice
        break

if indian_female_voice:
    engine.setProperty("voice", indian_female_voice.id)
else:
    print("No Indian Female voice found. Using default voice.")


model_name = "meta-llama/Llama-3.2-1b"
huggingface_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


recognizer = sr.Recognizer()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def save_database(data):
    """Save user data to the database"""
    serializable_data = {}
    for user_id, user_info in data.items():
        serializable_user_info = user_info.copy()
        if isinstance(serializable_user_info.get('face_encoding'), np.ndarray):
            serializable_user_info['face_encoding'] = serializable_user_info['face_encoding'].tolist()
        serializable_data[user_id] = serializable_user_info

    with open(DATABASE_FILE, 'w') as file:
        json.dump(serializable_data, file, indent=4)

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as file:
            data = json.load(file)
            for user_id, user_info in data.items():
                user_info['face_encoding'] = np.array(user_info['face_encoding'], dtype=float)
            return data
    return {}

def calculate_distance(descriptor1, descriptor2):
    """Calculate distance between two face descriptors"""
    return np.linalg.norm(descriptor1 - descriptor2)

def find_registered_user(face_encoding):
    """Find a registered user by comparing face descriptors"""
    database = load_database()
    for user_id, user_info in database.items():
        registered_descriptor = user_info.get('face_descriptor')
        if registered_descriptor is not None:
            if calculate_distance(face_descriptor, registered_descriptor) < 0.6:
                return user_info
    return None

def register_new_user(face_encoding):
    """Register a new user"""
    talk("I'll help you register. Please answer the following questions.")
    
    talk("What's your name?")
    name = listen().strip().capitalize()
    
    # talk(f"Nice to meet you, {name}! What nickname would you like me to call you?")
    # nickname = listen().strip()
    
    talk(f"How old are you, {name}?")
    age = listen().strip()
    
    talk("What's your phone number?")
    phone = listen().strip()

    user_id = f"{name}_{phone}"
    database = load_database()
    database[user_id] = {
        'name': name,
        #'nickname': nickname,
        'age': age,
        'phone': phone,
        'face_encoding': face_encoding
    }
    save_database(database)
    
    talk(f"Thanks, {name}! I've saved your details.")
    return database[user_id]

def talk(text):
    """Text-to-speech function"""
    print(f"Supram: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    """Speech recognition function with retry mechanism"""
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    talk("Sorry, I didn't catch that. Could you please repeat?")
                    continue  # Retry listening
                except sr.RequestError:
                    talk("There was an error with the speech recognition service.")
                    return ""
            except sr.WaitTimeoutError:
                talk("No speech detected. Please try speaking again.")
                continue  # Retry listening
            
            
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def generate_response(prompt):
    """Generate AI response"""
    
    if not prompt.strip():  # Check for empty input
        return "I didn't receive a clear input. Could you please speak again?"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def fetch_internet_response(query):
    """Fetch and sort internet search results by recency"""
    try:
        url = f"https://www.bing.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all search result items
        results = soup.find_all("li", class_="b_algo")
        
        # Create a list to store results with their dates
        sorted_results = []
        
        for result in results:
            # Try to find the date
            date_tag = result.find("span", class_="sc_date")
            date_str = date_tag.text.strip() if date_tag else None
            
            snippet_tag = result.find("p")
            snippet = snippet_tag.text.strip() if snippet_tag else ""
            
            # If no date found, put it at the end of sorting
            try:
                # You might need to adjust date parsing based on actual format
                if date_str:
                    parsed_date = dateparser.parse(date_str)
                else:
                    parsed_date = datetime.min
                
                sorted_results.append({
                    'date': parsed_date,
                    'content': snippet
                })
            except:
                # Fallback if date parsing fails
                sorted_results.append({
                    'date': datetime.min,
                    'content': snippet
                })
        
        # Sort results by date in descending order
        sorted_results.sort(key=lambda x: x['date'], reverse=False)
        
        # Extract and format top 5 most recent results
        formatted_results = [result['content'] for result in sorted_results[:5]]
        
        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error fetching internet data: {str(e)}"

def open_application(command):
    """Open specific applications or perform actions"""
    try:
        if "whatsapp" in command:
            subprocess.Popen([r"C:\Users\Ram\OneDrive - IIT Hyderabad\Windows\Program Files\WindowsApps\5319275A.WhatsAppDesktop_2.2313.6.0_x64__cv1g1gvanyjgm"])
            talk("Opening WhatsApp.")
        elif "youtube" in command and "brave" in command:
            query = command.replace("open YouTube in brave and play", "").strip()
            brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
            if os.path.exists(brave_path):
                subprocess.Popen([brave_path, f"https://www.youtube.com/results?search_query={query}"])
                talk(f"Opening YouTube in Brave and playing {query}.")
            else:
                talk("Brave browser is not installed.")
        elif "brave" in command:
            subprocess.Popen(["C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"])
            talk("Opening Brave browser.")
        elif "youtube" in command:
            query = command.replace("play", "").strip()
            url = f"https://www.youtube.com/results?search_query={query}"
            webbrowser.open(url)
            talk(f"Playing {query} on YouTube.")
        else:
            talk("I can't open that application right now.")
    except Exception as e:
        talk(f"Error opening application: {e}")

def generate_frames():
    """Generate video frames for the live feed"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab a frame from the camera.")
                break

            frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"Error during frame generation: {e}")
            break

    cap.release()
    print("Camera released.")

def assistant():
    """Main assistant function with advanced face recognition"""
    talk("Hello! I'm ready to help. Please show your face to the camera.")

    # Initialize face recognition
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        talk("I couldn't access the camera. Please check your setup.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            talk("I couldn't capture a frame. Please try again.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                landmarks = predictor(gray, face)
                face_descriptor = face_recognition_model.compute_face_descriptor(frame, landmarks)

                # Try to recognize the user
                user = find_registered_user(np.array(face_descriptor))

                if user:
                    nickname = user.get('nickname', user.get('nickname', 'Friend'))
                    talk(f"Welcome back, {nickname}!")
                    user_details = user
                else:
                    # Register new user
                    user_details = register_new_user(np.array(face_descriptor))

                # Voice interaction loop
                while True:
                    user_input = listen()
                    
                    if "exit" in user_input.lower():
                        talk("Goodbye! See you later!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
                    if not user_input:
                        continue;
                    
                
                    if "search" in user_input.lower():
                        query = user_input.replace("search", "").strip()
                        internet_result = fetch_internet_response(query)
                        talk(internet_result)
                    elif "open" in user_input.lower():
                        open_application(user_input.lower())
                    else:
                        response = generate_response(user_input)
                        talk(response)

        cv2.imshow("Supram Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask routes and other setup remain the same as in the original code

if __name__ == "__main__":
    from threading import Thread

    def run_flask():
        app.run(port=5000, debug=False, threaded=True)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    assistant()