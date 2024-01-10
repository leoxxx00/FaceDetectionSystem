import cv2
import face_recognition

# Load your known images and encode them
known_image1 = face_recognition.load_image_file("1.jpg")
known_encoding1 = face_recognition.face_encodings(known_image1)[0]

known_image2 = face_recognition.load_image_file("2.jpg")
known_encoding2 = face_recognition.face_encodings(known_image2)[0]

# Create an array of known face encodings and corresponding names
known_face_encodings = [known_encoding1, known_encoding2]
known_face_names = ["Person 1", "Person 2"]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Set webcam resolution
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces with a tolerance of 0.6
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
