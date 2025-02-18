import face_recognition
import cv2
import numpy as np
import time

# Function to validate name (no numbers allowed)
def validate_name(name):
    return name.isalpha()

# Function to validate age (only numbers allowed)
def validate_age(age):
    return age.isdigit()

# Function to validate phone number (only numbers allowed, exactly 10 digits)
def validate_phone(phone):
    return phone.isdigit() and len(phone) == 10

# Function to save user details
def save_user_details(name, age, phone, country_code, face_encoding):
    user_details = {
        "name": name,
        "age": age,
        "phone": f"+{country_code}{phone}",
        "face_encoding": face_encoding
    }
    return user_details

# Load known faces and their details
known_faces = []
known_details = []

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Confidence threshold for face recognition (0.6 is a good starting point)
confidence_threshold = 0.6

# Delay before registering a new face (in seconds)
new_face_delay = 5

# Dictionary to track the time when a new face is first detected
new_face_detected_times = {}

# Dictionary to track the last known name for each face
face_id_to_name = {}

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Reset the list of detected faces in the current frame
    detected_faces = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Generate a unique face ID based on the face location
        face_id = f"{top}-{right}-{bottom}-{left}"

        # Check if the face matches any known face
        matches = face_recognition.compare_faces([face["face_encoding"] for face in known_details], face_encoding)
        face_distances = face_recognition.face_distance([face["face_encoding"] for face in known_details], face_encoding)

        # Get the best match (lowest distance) and check if it meets the confidence threshold
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        name = "Unknown"
        details = ""

        if best_match_index is not None and face_distances[best_match_index] <= confidence_threshold:
            # Face is recognized
            name = known_details[best_match_index]["name"]
            details = f"Name: {name}, Age: {known_details[best_match_index]['age']}, Phone: {known_details[best_match_index]['phone']}"
            face_id_to_name[face_id] = name  # Update the face ID to name mapping
        else:
            # Face is not recognized
            if face_id not in new_face_detected_times:
                # Start the timer for new face detection
                new_face_detected_times[face_id] = time.time()
            elif time.time() - new_face_detected_times[face_id] >= new_face_delay:
                # Ask for user details if the face is unknown and the delay has passed
                print("New face detected. Please provide the following details:")
                while True:
                    name = input("Enter your name (no numbers allowed): ")
                    if validate_name(name):
                        break
                    print("Invalid name. Please try again.")

                while True:
                    age = input("Enter your age (no alphabets allowed): ")
                    if validate_age(age):
                        break
                    print("Invalid age. Please try again.")

                while True:
                    country_code = input("Enter your country code (e.g., 91 for India): ")
                    if country_code.isdigit():
                        break
                    print("Invalid country code. Please try again.")

                while True:
                    phone = input("Enter your phone number (10 digits, no alphabets): ")
                    if validate_phone(phone):
                        break
                    print("Invalid phone number. Please try again.")

                # Save the new user details
                user_details = save_user_details(name, age, phone, country_code, face_encoding)
                known_details.append(user_details)
                details = f"Name: {name}, Age: {age}, Phone: +{country_code}{phone}"
                face_id_to_name[face_id] = name  # Update the face ID to name mapping
                del new_face_detected_times[face_id]  # Remove the face from the tracking dictionary

        # Add the detected face to the list
        detected_faces.append((name, details, (top, right, bottom, left)))

    # Clear the tracking dictionary for faces that are no longer in the frame
    for face_id in list(new_face_detected_times.keys()):
        if face_id not in [f"{top}-{right}-{bottom}-{left}" for (top, right, bottom, left) in face_locations]:
            del new_face_detected_times[face_id]

    # Display the results for all detected faces
    for name, details, (top, right, bottom, left) in detected_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name and details below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, details, (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()