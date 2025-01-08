import cv2
import numpy as np
import mediapipe as mp

# Inicialización de Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables para dibujo
drawing_mode = False
last_x, last_y = None, None
canvas = None

# Función para contar dedos levantados
def count_fingers(hand_landmarks):
    fingers = []
    # Coordenadas relevantes para los dedos
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_dips = [mp_hands.HandLandmark.INDEX_FINGER_DIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                   mp_hands.HandLandmark.RING_FINGER_DIP,
                   mp_hands.HandLandmark.PINKY_DIP]

    for tip, dip in zip(finger_tips, finger_dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Pulgar
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen horizontalmente para efecto de espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Inicializar lienzo si no está creado
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convertir la imagen a RGB para Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Contar dedos levantados
            fingers = count_fingers(hand_landmarks)
            total_fingers = sum(fingers)

            # Detectar gestos
            if total_fingers == 5:
                drawing_mode = True
                last_x, last_y = None, None
                cv2.putText(frame, "Modo dibujo activado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif total_fingers == 0:
                drawing_mode = False
                last_x, last_y = None, None
                cv2.putText(frame, "Modo dibujo desactivado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Si un dedo está levantado y el modo de dibujo está activado
            if total_fingers == 1 and drawing_mode:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                if last_x is not None and last_y is not None:
                    # Dibujar en el lienzo
                    cv2.line(canvas, (last_x, last_y), (x, y), (255, 0, 0), 5)
                last_x, last_y = x, y

    # Combinar lienzo con la imagen de la cámara
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostrar la imagen
    cv2.imshow('Dibujo en vivo', combined_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
