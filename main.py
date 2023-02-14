import cv2
import hand_detection

if __name__ == '__main__':
    rece = hand_detection.HandDetection()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()

        rece.hand_detecting(image)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
