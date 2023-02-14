import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandDetection:
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_x = 0
        self.hand_y = 0
        self.results = None
        self.hand_closed = False

    def hand_detecting(self, image):
        self.hand_closed = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                x, y = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y

                self.hand_x = int(x * 1280)#SCREEN_WIDTH)
                self.hand_y = int(y * 720)#SCREEN_HEIGHT)

                x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                x3, y3 = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
                x4, y4 = hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y
                x5, y5 = hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y

                if y3 > y:
                    if y4 > y and y5 > y and y2 < y:
                        print("JEDENNN")
                    else:
                        self.hand_closed = True
                        print("ZAMKNIETAAA")
                else:
                    print("OTWARTAAA")

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        return image


