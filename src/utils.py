"""
utils.py — Funciones auxiliares compartidas por todos los módulos del proyecto.

Incluye: carga de imágenes desde carpeta, redimensionado conservando aspecto,
dibujo de etiqueta sobre imagen y decorador para medir tiempo de ejecución.
"""

import os
import time
import functools

import cv2
import numpy as np


def load_images_from_folder(path: str) -> dict[str, np.ndarray]:
    """
    Carga todas las imágenes de una carpeta y las devuelve en un diccionario.

    La clave es el nombre del archivo sin extensión (nombre de clase).
    Omite archivos que no sean imágenes reconocibles por OpenCV.

    Args:
        path: Ruta a la carpeta con las imágenes.

    Returns:
        Diccionario {nombre_clase: imagen_bgr}.

    Raises:
        FileNotFoundError: Si la carpeta no existe.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"La carpeta no existe: {path}")

    imagenes = {}
    extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for nombre_archivo in sorted(os.listdir(path)):
        nombre_base, extension = os.path.splitext(nombre_archivo)
        if extension.lower() not in extensiones_validas:
            continue

        ruta_completa = os.path.join(path, nombre_archivo)
        img = cv2.imread(ruta_completa)

        if img is None:
            print(f"[utils] Advertencia: no se pudo leer '{ruta_completa}', omitiendo.")
            continue

        imagenes[nombre_base] = img

    return imagenes


def resize_keeping_aspect(img: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """
    Redimensiona una imagen conservando la relación de aspecto.

    El lado más largo queda igual a max_dim; el otro se escala proporcionalmente.

    Args:
        img:     Imagen de entrada (NumPy array BGR).
        max_dim: Tamaño máximo del lado más largo, en píxeles.

    Returns:
        Imagen redimensionada.
    """
    alto, ancho = img.shape[:2]
    lado_mayor = max(alto, ancho)

    if lado_mayor <= max_dim:
        return img

    escala = max_dim / lado_mayor
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    return cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def draw_label(img: np.ndarray, text: str, pos: tuple[int, int] = (50, 50)) -> None:
    """
    Dibuja una etiqueta de texto sobre la imagen (in-place).

    Usa un fondo oscuro semitransparente para mejorar la legibilidad.

    Args:
        img:  Imagen BGR donde se dibuja.
        text: Texto a mostrar.
        pos:  Esquina superior-izquierda del texto (x, y).
    """
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.0
    grosor = 2
    color_texto = (0, 255, 0)      # verde
    color_fondo = (0, 0, 0)        # negro

    (ancho_texto, alto_texto), baseline = cv2.getTextSize(text, fuente, escala, grosor)
    x, y = pos

    # Rectángulo de fondo
    cv2.rectangle(
        img,
        (x - 4, y - alto_texto - 4),
        (x + ancho_texto + 4, y + baseline + 4),
        color_fondo,
        cv2.FILLED,
    )
    # Texto encima
    cv2.putText(img, text, (x, y), fuente, escala, color_texto, grosor, cv2.LINE_AA)


def timed(func):
    """
    Decorador que mide el tiempo de ejecución de una función en milisegundos.

    El tiempo se almacena en el atributo 'last_ms' del objeto devuelto.
    Para funciones sueltas, imprime el resultado en consola.

    Uso:
        @timed
        def mi_funcion():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        wrapper.last_ms = (fin - inicio) * 1000
        return resultado

    wrapper.last_ms = 0.0
    return wrapper
