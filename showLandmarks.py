import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Shows the face landmark map
def faceMarkMap(image, faces):
    for face in faces.multi_face_landmarks:
        h, w, c = image.shape
        for i, lm in enumerate(face.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 2, (0,0,255), -1)
            cv2.putText(image, str(i), (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)

# Shows the hand landmark map
def handMarkMap(image, hand):
    mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
