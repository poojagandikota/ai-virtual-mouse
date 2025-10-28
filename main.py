import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe and camera
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

index_y = 0

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Index finger tip landmark (id=8)
                if id == 8:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    index_x = int(lm.x * screen_w)
                    index_y = int(lm.y * screen_h)

                # Thumb tip landmark (id=4)
                if id == 4:
                    thumb_x = int(lm.x * screen_w)
                    thumb_y = int(lm.y * screen_h)

                    # Check if thumb and index finger are close (for click action)
                    if abs(index_y - thumb_y) < 40:
                        pyautogui.click()
                        pyautogui.sleep(0.2)

            # Move cursor according to index finger position
            pyautogui.moveTo(index_x, index_y, duration=0.1)

    cv2.imshow("AI Virtual Mouse", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
