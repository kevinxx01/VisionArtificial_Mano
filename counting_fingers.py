import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

def detect_fingers(hand_landmarks, width, height):
    thumb_points = [1, 2, 4]
    palm_points = [0, 1, 2, 5, 9, 13, 17]
    fingertips_points = [8, 12, 16, 20]
    finger_base_points = [6, 10, 14, 18]

    coordinates_thumb = []
    coordinates_palm = []
    coordinates_ft = []
    coordinates_fb = []

    for index in thumb_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_thumb.append([x, y])

    for index in palm_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_palm.append([x, y])

    for index in fingertips_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_ft.append([x, y])

    for index in finger_base_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_fb.append([x, y])

    return coordinates_thumb, coordinates_palm, coordinates_ft, coordinates_fb

def calculate_thumb_finger_angle(coordinates_thumb):
    p1 = np.array(coordinates_thumb[0])
    p2 = np.array(coordinates_thumb[1])
    p3 = np.array(coordinates_thumb[2])

    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    thumb_finger = angle > 150
    return thumb_finger

def draw_info_on_frame(frame, fingers, thickness, fingers_counter):
    NEON_PINK = (255, 55, 255)
    NEON_PURPLE = (214, 60, 242)
    NEON_GREEN = (50, 255, 50)
    NEON_ORANGE = (255, 153, 18)
    NEON_BLUE = (18, 216, 255)

    cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
    cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)

    colors = [NEON_PINK, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, NEON_BLUE]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    for i, (color, name) in enumerate(zip(colors, finger_names)):
        cv2.rectangle(frame, (100 + i * 60, 10), (150 + i * 60, 60), color, thickness[i])
        cv2.putText(frame, name, (100 + i * 60, 80), 1, 1, (255, 255, 255), 2)

    return frame

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            fingers_counter = "_"
            thickness = [2, 2, 2, 2, 2]
            fingers = [False, False, False, False, False]  # Initialize fingers list

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Only process the first detected hand
                coordinates_thumb, coordinates_palm, coordinates_ft, coordinates_fb = detect_fingers(hand_landmarks, width, height)
                thumb_finger = calculate_thumb_finger_angle(coordinates_thumb)

                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                d_centroid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centroid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centroid_ft - d_centroid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counter = str(np.count_nonzero(fingers == True))

                for i, finger in enumerate(fingers):
                    if finger:
                        thickness[i] = -1

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            frame = draw_info_on_frame(frame, fingers, thickness, fingers_counter)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





