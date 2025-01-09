# Dibujopiton

## Description
Dibujopiton es un programa en Python que permite dibujar en tiempo real e interactuar mediante gestos de mano detectados a través de una cámara web. Incluye herramientas como selección de colores, dibujo de formas, ajuste del grosor de línea y efectos 3D en los dibujos.

## Features
- **Detección de gestos de mano**: Reconoce gestos para habilitar o deshabilitar el modo de dibujo y para interactuar con botones virtuales.
- **Herramientas de dibujo**:
  - Dibujo a mano alzada usando un solo dedo levantado.
  - Dibujo de formas predefinidas (rectángulos y círculos).
  - Ajuste del grosor de línea.
  - Borrado del lienzo.
- **Selección de colores**: Cambia entre rojo, verde y azul para dibujar.
- **Efectos 3D**:
  - Aplica transformaciones de perspectiva para simular rotaciones 3D.
  - Añade sombreado en relieve para mejorar los dibujos.
- **Botones interactivos**: Botones visuales en la pantalla para facilitar la interacción.

## How It Works
1. El programa captura video desde la cámara web usando OpenCV.
2. La detección de manos se realiza en el espacio HSV basado en el color de la piel.
3. Se usan el "convex hull" y los defectos de convexidad para detectar dedos y gestos.
4. Los gestos controlan los modos de dibujo y otras funcionalidades.
5. El dibujo se muestra en un lienzo superpuesto combinado con la transmisión en vivo de la cámara.
6. Opcionalmente, se pueden aplicar transformaciones y efectos 3D al dibujo.

## Prerequisites
- Python 3.7+
- OpenCV 4.5+
- NumPy

Instala las dependencias usando pip:
```bash
pip install opencv-python numpy
```

## Usage
1. Ejecuta el script:
   ```bash
   python hand_draw_3d.py
   ```
2. La transmisión de la cámara web se abrirá en una ventana titulada "Dibujo en vivo."
3. Usa gestos para interactuar con el programa:
   - Levanta los cinco dedos para activar el modo de dibujo.
   - Baja todos los dedos para desactivar el modo de dibujo.
   - Levanta un dedo para dibujar a mano alzada.
   - Interactúa con los botones haciendo clic en ellos con el ratón.
4. Presiona `q` para salir.

## Key Interactions
- **Botones de colores**:
  - Rojo: Esquina superior izquierda (10, 10) a (150, 150)
  - Verde: Al lado del botón rojo
  - Azul: Al lado del botón verde
- **Otros botones**:
  - Borrar: Limpia el lienzo.
  - 3D: Activa o desactiva el modo de perspectiva 3D.
  - Relieve: Añade sombreado en relieve a los dibujos 3D (solo disponible en modo 3D).
  - +Gros: Aumenta el grosor de la línea.
  - -Gros: Disminuye el grosor de la línea.
  - Rect: Cambia al modo de forma rectangular.
  - Circ: Cambia al modo de forma circular.

## Notes
- Ajusta el rango de color de la piel en `detect_fingers()` para adaptarlo a tus condiciones de iluminación.
- Si el lienzo no coincide con el tamaño del video, se reinicializará automáticamente.

## Troubleshooting
- **Problemas con la detección de gestos**: Asegúrate de tener buena iluminación y un fondo que contraste con el tono de tu piel.
- **Retrasos**: Reduce la resolución del video o simplifica los efectos 3D.

## License
Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Acknowledgments
Un agradecimiento especial a las comunidades de OpenCV y NumPy por sus increíbles herramientas y bibliotecas.

