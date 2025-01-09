import cv2
import numpy as np

# Variables para dibujo
drawing_mode = False
last_x, last_y = None, None
canvas = None
current_color = (255, 0, 0)  # Color inicial: Azul
mouse_click_position = None  # Almacena la posición del clic del mouse
render_3d_mode = False  # Indica si el modo 3D está activado
add_3d_relief = False  # Indica si se debe aplicar relieve 3D a las líneas
rotation_angle_x = 0  # Ángulo de rotación en X para el 3D
rotation_angle_y = 0  # Ángulo de rotación en Y para el 3D
mouse_drag_start = None  # Posición inicial para arrastrar el mouse

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

# Función de callback para manejar clics y arrastres del mouse
def mouse_callback(event, x, y, flags, param):
    global mouse_click_position, mouse_drag_start, rotation_angle_x, rotation_angle_y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_position = (x, y)
        mouse_drag_start = (x, y)  # Iniciar arrastre
        print(f"Clic detectado en: {mouse_click_position}")

    elif event == cv2.EVENT_MOUSEMOVE and mouse_drag_start:
        # Calcular diferencias de movimiento para rotar
        dx = x - mouse_drag_start[0]
        dy = y - mouse_drag_start[1]
        rotation_angle_x += dy * 0.5  # Ajustar sensibilidad
        rotation_angle_y += dx * 0.5
        mouse_drag_start = (x, y)  # Actualizar posición inicial

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_drag_start = None  # Terminar arrastre

# Función para aplicar el efecto 3D al dibujo con rotación
def render_3d(canvas):
    rows, cols, _ = canvas.shape
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    # Ajustar puntos de destino según la rotación
    angle_x_rad = np.radians(rotation_angle_x)
    angle_y_rad = np.radians(rotation_angle_y)

    dx = np.tan(angle_y_rad) * cols / 2
    dy = np.tan(angle_x_rad) * rows / 2

    dst_points = np.float32([
        [50 + dx, 50 + dy],
        [cols - 50 - dx, 30 + dy],
        [70 + dx, rows - 50 - dy],
        [cols - 70 - dx, rows - 70 - dy]
    ])

    # Crear una transformación de perspectiva
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(canvas, matrix, (cols, rows))

# Función para agregar relieve 3D a las líneas
def add_relief(canvas):
    rows, cols, _ = canvas.shape
    relief_canvas = np.zeros_like(canvas)

    # Aplicar un efecto de sombreado para simular relieve
    for offset in range(1, 11):  # Incrementar el rango para mayor relieve
        shifted_canvas = np.roll(canvas, offset, axis=0)  # Desplazamiento en Y
        shifted_canvas = np.roll(shifted_canvas, offset, axis=1)  # Desplazamiento en X
        relief_canvas = cv2.addWeighted(relief_canvas, 1.0, shifted_canvas, 0.3, 0)

    return cv2.add(canvas, relief_canvas)

# Captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Dibujo en vivo')
cv2.setMouseCallback('Dibujo en vivo', mouse_callback)

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

    # Crear botones para cambiar colores y activar el modo 3D
    cv2.rectangle(frame, (10, 10), (150, 150), (0, 0, 255), -1)  # Rojo
    cv2.rectangle(frame, (160, 10), (300, 150), (0, 255, 0), -1)  # Verde
    cv2.rectangle(frame, (310, 10), (450, 150), (255, 0, 0), -1)  # Azul
    cv2.rectangle(frame, (460, 10), (600, 150), (200, 200, 200), -1)  # Borrar
    cv2.rectangle(frame, (610, 10), (750, 150), (100, 100, 255), -1)  # 3D

    # Etiquetas de los botones
    cv2.putText(frame, "Rojo", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Verde", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Azul", (340, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Borrar", (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "3D", (660, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if render_3d_mode:
        # Mostrar botón adicional para activar el relieve 3D
        cv2.rectangle(frame, (760, 10), (900, 150), (150, 150, 255), -1)  # Relieve 3D
        cv2.putText(frame, "Relieve", (780, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detectar interacción con los botones mediante clic del mouse
    if mouse_click_position:
        x, y = mouse_click_position
        if 10 <= x <= 150 and 10 <= y <= 150:  # Rojo
            current_color = (0, 0, 255)
            print("Color cambiado a Rojo")
        elif 160 <= x <= 300 and 10 <= y <= 150:  # Verde
            current_color = (0, 255, 0)
            print("Color cambiado a Verde")
        elif 310 <= x <= 450 and 10 <= y <= 150:  # Azul
            current_color = (255, 0, 0)
            print("Color cambiado a Azul")
        elif 460 <= x <= 600 and 10 <= y <= 150:  # Borrar
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            print("Canvas borrado")
        elif 610 <= x <= 750 and 10 <= y <= 150:  # Activar modo 3D
            render_3d_mode = not render_3d_mode
            print(f"Modo 3D {'activado' if render_3d_mode else 'desactivado'}")
        elif render_3d_mode and 760 <= x <= 900 and 10 <= y <= 150:  # Activar relieve 3D
            add_3d_relief = not add_3d_relief
            print(f"Relieve 3D {'activado' if add_3d_relief else 'desactivado'}")
        mouse_click_position = None  # Reset posición clickeada

    # Aplicar modo 3D si está activado
    if render_3d_mode:
        transformed_canvas = render_3d(canvas)
        if add_3d_relief:
            transformed_canvas = add_relief(transformed_canvas)
        combined_frame = cv2.addWeighted(frame, 0.5, transformed_canvas, 0.5, 0)
    else:
        combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostrar la imagen
    cv2.imshow('Dibujo en vivo', combined_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
