import cv2
import numpy as np

# Variables para dibujo
drawing_mode = False
last_x, last_y = None, None
canvas = None
current_color = (255, 0, 0)  # Color inicial: Azul

# Función para detectar la mano y los dedos
def detect_fingers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de color para la piel (ajustable según la iluminación)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Crear máscara para el color de la piel
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar suavizado a la máscara
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, []

    # Contorno más grande (asumimos que es la mano)
    max_contour = max(contours, key=cv2.contourArea)

    # Encontrar hull convexo
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    if defects is None:
        return max_contour, []

    # Contar defectos de convexidad como dedos
    fingers = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        # Calcular ángulo entre los puntos
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

        # Contar como un dedo si el ángulo es pequeño
        if angle <= np.pi / 2 and d > 10000:
            fingers += 1

    # Considerar el pulgar como un dedo extra
    fingers += 1

    return max_contour, fingers

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

    # Detectar dedos
    contour, fingers = detect_fingers(frame)

    if contour is not None:
        # Dibujar contorno de la mano
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # Detectar gestos
        if fingers == 5:
            drawing_mode = True
            last_x, last_y = None, None
            cv2.putText(frame, "Modo dibujo activado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif fingers == 0:
            drawing_mode = False
            last_x, last_y = None, None
            cv2.putText(frame, "Modo dibujo desactivado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Si un dedo está levantado y el modo de dibujo está activado
        if fingers == 1 and drawing_mode:
            hull = cv2.convexHull(contour)
            moments = cv2.moments(hull)
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])

                if last_x is not None and last_y is not None:
                    # Dibujar en el lienzo
                    cv2.line(canvas, (last_x, last_y), (x, y), current_color, 5)
                last_x, last_y = x, y

    # Crear botones para cambiar colores
    cv2.rectangle(frame, (10, 10), (60, 60), (0, 0, 255), -1)  # Rojo
    cv2.rectangle(frame, (70, 10), (120, 60), (0, 255, 0), -1)  # Verde
    cv2.rectangle(frame, (130, 10), (180, 60), (255, 0, 0), -1)  # Azul
    cv2.rectangle(frame, (190, 10), (240, 60), (200, 200, 200), -1)  # Borrar

    # Detectar clics en los botones
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_buttons = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
    contours_buttons, _ = cv2.findContours(mask_buttons, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_buttons:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 <= x <= 60 and 10 <= y <= 60:  # Rojo
            current_color = (0, 0, 255)
        elif 70 <= x <= 120 and 10 <= y <= 60:  # Verde
            current_color = (0, 255, 0)
        elif 130 <= x <= 180 and 10 <= y <= 60:  # Azul
            current_color = (255, 0, 0)
        elif 190 <= x <= 240 and 10 <= y <= 60:  # Borrar
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Combinar lienzo con la imagen de la cámara
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostrar la imagen
    cv2.imshow('Dibujo en vivo', combined_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
