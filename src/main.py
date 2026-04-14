"""
main.py — Punto de entrada principal del proyecto P2_Fernando_Zea.

Modos de operación:
  image   → Clasifica una imagen fija y muestra el resultado anotado.
  webcam  → Clasificación en tiempo real desde la cámara web.
  benchmark → Evaluación comparativa de los 3 descriptores.

Descriptores disponibles: orb, sift, canny.

Ejemplos de uso:
    python src/main.py --mode image --descriptor orb --input data/tests/test1.jpg
    python src/main.py --mode webcam --descriptor sift
    python src/main.py --mode benchmark
"""

import argparse
import os
import sys

import cv2

# Añadir src/ al path para importar los módulos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from descriptors import ORBMatcher, SIFTMatcher, AKAZEMatcher
from classifier import find_id
from utils import load_images_from_folder, resize_keeping_aspect, draw_label


# ------------------------------------------------------------------
# Rutas por defecto
# ------------------------------------------------------------------
DIRECTORIO_BASE = os.path.join(os.path.dirname(__file__), "..")
RUTA_REFERENCIAS = os.path.join(DIRECTORIO_BASE, "data", "references")


def _crear_matcher(nombre_descriptor: str):
    """
    Instancia el matcher correspondiente al nombre dado.

    Args:
        nombre_descriptor: 'orb', 'sift' o 'canny' (insensible a mayúsculas).

    Returns:
        Instancia del matcher elegido.

    Raises:
        ValueError: Si el nombre no es reconocido.
    """
    nombre_lower = nombre_descriptor.lower()
    if nombre_lower == "orb":
        return ORBMatcher(nfeatures=1000)
    if nombre_lower == "sift":
        return SIFTMatcher()
    if nombre_lower == "akaze":
        return AKAZEMatcher()
    raise ValueError(f"Descriptor desconocido: '{nombre_descriptor}'. "
                     f"Opciones: orb, sift, akaze.")


def _cargar_y_preparar_referencias(matcher) -> bool:
    """
    Carga las imágenes de referencia y llama a precompute().

    Args:
        matcher: Instancia del matcher a preparar.

    Returns:
        True si la carga fue exitosa, False en caso contrario.
    """
    try:
        referencias = load_images_from_folder(RUTA_REFERENCIAS)
    except FileNotFoundError as e:
        print(f"[main] Error: {e}")
        return False

    if not referencias:
        print("[main] Error: no hay imágenes de referencia en data/references/")
        return False

    # Redimensionar para consistencia
    referencias = {k: resize_keeping_aspect(v, 640) for k, v in referencias.items()}
    matcher.precompute(referencias)
    print(f"[main] Clases cargadas: {matcher.class_names}")
    return True


# ------------------------------------------------------------------
# Modos de operación
# ------------------------------------------------------------------

def modo_imagen(descriptor: str, ruta_entrada: str) -> None:
    """
    Clasifica una imagen estática y muestra el resultado.

    Args:
        descriptor:    Nombre del descriptor a usar.
        ruta_entrada:  Ruta a la imagen de entrada.
    """
    if not os.path.isfile(ruta_entrada):
        print(f"[main] Error: no se encontró el archivo '{ruta_entrada}'")
        sys.exit(1)

    img = cv2.imread(ruta_entrada)
    if img is None:
        print(f"[main] Error: OpenCV no pudo leer '{ruta_entrada}'")
        sys.exit(1)

    matcher = _crear_matcher(descriptor)
    if not _cargar_y_preparar_referencias(matcher):
        sys.exit(1)

    img_redim = resize_keeping_aspect(img, 640)
    clase, scores = find_id(img_redim, matcher)

    if clase:
        etiqueta = f"{clase} ({max(scores):.2f})"
        print(f"[main] Detectado: {etiqueta}")
    else:
        etiqueta = "No identificado"
        print("[main] No se identificó ningún objeto conocido.")

    # Mostrar resultado
    img_resultado = img_redim.copy()
    draw_label(img_resultado, etiqueta)
    cv2.imshow(f"Resultado — {descriptor.upper()}", img_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def modo_webcam(descriptor: str) -> None:
    """
    Clasificación en tiempo real usando la cámara web.

    Muestra el nombre del objeto detectado con cv2.putText.
    Presionar 'q' para salir.

    Args:
        descriptor: Nombre del descriptor a usar.
    """
    matcher = _crear_matcher(descriptor)
    if not _cargar_y_preparar_referencias(matcher):
        sys.exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[main] Error: no se pudo abrir la cámara web (índice 0).")
        sys.exit(1)

    # Forzar formato MJPG para evitar problemas de conversión YUYV en Linux
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Descartar los primeros frames hasta que la cámara estabilice la exposición
    for _ in range(30):
        cap.read()

    nombre_ventana = f"Webcam - {descriptor.upper()}"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    print(f"[main] Modo webcam activo con {descriptor.upper()}. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[main] Error: no se pudo leer frame de la cámara.")
            break

        frame_redim = resize_keeping_aspect(frame, 640)
        clase, scores = find_id(frame_redim, matcher)

        if clase:
            etiqueta = f"{clase} ({max(scores):.2f})"
        else:
            etiqueta = "No identificado"

        # Dibujar etiqueta sobre el frame (igual que en el video de referencia)
        draw_label(frame_redim, etiqueta, pos=(20, 50))

        # Mostrar scores individuales por clase para diagnóstico
        if scores and matcher.class_names:
            for i, (nombre, score) in enumerate(zip(matcher.class_names, scores)):
                debug_texto = f"{nombre}: {score:.2f} / {matcher.threshold}"
                cv2.putText(
                    frame_redim,
                    debug_texto,
                    (20, 90 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow(nombre_ventana, frame_redim)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def modo_benchmark() -> None:
    """Delega en el módulo benchmark.py."""
    import benchmark
    benchmark.correr_benchmark()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P2 Visión Computacional — Identificación de objetos con descriptores OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python src/main.py --mode image --descriptor orb --input data/tests/test1.jpg\n"
            "  python src/main.py --mode webcam --descriptor sift\n"
            "  python src/main.py --mode benchmark\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["image", "webcam", "benchmark"],
        required=True,
        help="Modo de operación.",
    )
    parser.add_argument(
        "--descriptor",
        choices=["orb", "sift", "akaze"],
        default="orb",
        help="Descriptor a usar (solo para --mode image y --mode webcam).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Ruta a la imagen de entrada (obligatorio con --mode image).",
    )
    return parser


def main() -> None:
    parser = _construir_parser()
    args = parser.parse_args()

    if args.mode == "image":
        if not args.input:
            parser.error("--input es obligatorio con --mode image.")
        modo_imagen(args.descriptor, args.input)

    elif args.mode == "webcam":
        modo_webcam(args.descriptor)

    elif args.mode == "benchmark":
        modo_benchmark()


if __name__ == "__main__":
    main()
