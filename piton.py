import cv2
import numpy as np

# Variables para dibujo
drawing_mode = False
last_x, last_y = None, None
canvas = None
current_color = (255, 0, 0)  # Color inicial: Azul
calibration_step = 0
calibrated = False
lower_skin = None
upper_skin = None

# Función para realizar calibración de la piel en pasos
def calibrate_skin(frame):
    global lower_skin, upper_skin, calibration_step, calibrated
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w, _ = frame.shape
    # Definir regiones para cada paso de calibración
    if calibration_step == 0:
        # Calibrar con toda la mano
        x1, y1, x2, y2 = w // 2 - 100, h // 2 - 100, w // 2 + 100, h // 2 + 100
        cv2.putText(frame, "Coloca toda la mano", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    elif calibration_step == 1:
        # Calibrar con el dedo índice
        x1, y1, x2, y2 = w // 2 - 50, h // 2 - 50, w // 2 + 50, h // 2 + 50
        cv2.putText(frame, "Coloca solo un dedo", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        calibrated = True
        cv2.putText(frame, "Calibración completada", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return

    roi = hsv[y1:y2, x1:x2]
    # Obtener los valores mínimos y máximos de HSV en la región
    if calibration_step == 0:
        lower_skin = np.percentile(roi.reshape(-1, 3), 5, axis=0).astype(np.uint8)
        upper_skin = np.percentile(roi.reshape(-1, 3), 95, axis=0).astype(np.uint8)
    calibration_step += 1

    # Dibujar región de calibración
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Función para detectar la mano y los dedos
def detect_fingers(frame):
    global lower_skin, upper_skin
    if not calibrated:
        return None, []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

# Función para detectar la piel dentro de un área específica
def detect_skin_in_area(frame, x1, y1, x2, y2):
    global lower_skin, upper_skin
    if not calibrated:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Recortar el área y detectar piel
    roi = hsv[y1:y2, x1:x2]
    mask = cv2.inRange(roi, lower_skin, upper_skin)

    # Calcular si hay piel presente
    skin_area = cv2.countNonZero(mask)
    total_area = (x2 - x1) * (y2 - y1)

    return skin_area > 0.2 * total_area  # Activar si más del 20% es piel

# Captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen horizontalmente para efecto de espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Inicializar lienzo si no está creado o si el tamaño cambia
    if canvas is None or canvas.shape[:2] != (h, w):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Calibración inicial por pasos
    if not calibrated:
        calibrate_skin(frame)
        cv2.imshow('Dibujo en vivo', frame)
        key = cv2.waitKey(3000)  # Esperar 3 segundos entre pasos
        if key & 0xFF == ord('q'):
            break
        continue

    # Detectar dedos
    contour, fingers = detect_fingers(frame)

    finger_tip = None  # Posición de la punta del dedo índice

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
        if fingers == 1:
            hull = cv2.convexHull(contour)
            moments = cv2.moments(hull)
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                finger_tip = (x, y)

                if drawing_mode:
                    if last_x is not None and last_y is not None:
                        # Dibujar en el lienzo
                        cv2.line(canvas, (last_x, last_y), (x, y), current_color, 5)
                    last_x, last_y = x, y

    # Crear botones para cambiar colores (agrandados con hitbox extendida)
    cv2.rectangle(frame, (10, 10), (150, 150), (0, 0, 255), -1)  # Rojo
    cv2.rectangle(frame, (160, 10), (300, 150), (0, 255, 0), -1)  # Verde
    cv2.rectangle(frame, (310, 10), (450, 150), (255, 0, 0), -1)  # Azul
    cv2.rectangle(frame, (460, 10), (600, 150), (200, 200, 200), -1)  # Borrar

    # Detectar interacción con los botones mediante detección de piel
    if detect_skin_in_area(frame, 10, 10, 150, 150):  # Rojo
        current_color = (0, 0, 255)
    elif detect_skin_in_area(frame, 160, 10, 300, 150):  # Verde
        current_color = (0, 255, 0)
    elif detect_skin_in_area(frame, 310, 10, 450, 150):  # Azul
        current_color = (255, 0, 0)
    elif detect_skin_in_area(frame, 460, 10, 600, 150):  # Borrar
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
