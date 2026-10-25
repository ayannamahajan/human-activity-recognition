import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands()

label = input("Which activity are you recording? (standing/walking/jumping/sitting/waving): ")

csv_file = open('activity_data.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)

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
        frame, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=6),   
        mp_drawing.DrawingSpec(color=(143, 188, 143), thickness=2)                    
    )

     row = []
     for lm in results.pose_landmarks.landmark:
             row.extend([lm.x, lm.y, lm.z])
     row.append(label)
     csv_writer.writerow(row)   

    if hand_results.multi_hand_landmarks:            
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 225, 255), thickness=2, circle_radius=5),
                mp_drawing.DrawingSpec(color=(143, 188, 143), thickness=2)
            )

    cv2.imshow('HAR- Pose Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()



