import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import ImageFont, ImageDraw, Image

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()

model = joblib.load('activity_model.pkl')
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/SnellRoundhand.ttc", 60, index=1)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=6),
            mp_drawing.DrawingSpec(color=(143, 188, 143), thickness=2)
        )

        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        
        prediction = model.predict([row])[0]

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=5),
                mp_drawing.DrawingSpec(color=(143, 188, 143), thickness=2)            
            )

        if prediction:
         frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
         draw = ImageDraw.Draw(frame_pil)
         draw.text((20, 20), f'Activity: {prediction}', font=font, fill=(255, 255, 0))
         frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('HAR - Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()