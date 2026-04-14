"""
benchmark.py — Evaluación comparativa de los tres descriptores.

Carga las referencias y las imágenes de prueba, corre ORB, SIFT y Canny sobre
cada imagen de prueba, mide tiempos y genera:
  - results.csv con columnas: test_image, descriptor, predicted, true_label,
    best_score, time_ms, correct
  - Imágenes anotadas en output/ (keypoints/matches para ORB y SIFT,
    mapas de bordes para Canny)
  - Tabla resumen en consola

Uso:
    python src/benchmark.py
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

# Añadir src/ al path para importar los módulos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from descriptors import ORBMatcher, SIFTMatcher, AKAZEMatcher
from classifier import find_id
from utils import load_images_from_folder, resize_keeping_aspect


# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------
DIRECTORIO_BASE = os.path.join(os.path.dirname(__file__), "..")
RUTA_REFERENCIAS = os.path.join(DIRECTORIO_BASE, "data", "references")
RUTA_PRUEBAS = os.path.join(DIRECTORIO_BASE, "data", "tests")
RUTA_SALIDA = os.path.join(DIRECTORIO_BASE, "output")
RUTA_CSV = os.path.join(DIRECTORIO_BASE, "results.csv")


def _guardar_imagen_anotada_keypoints(
    img: np.ndarray,
    detector,
    nombre_descriptor: str,
    nombre_prueba: str,
) -> None:
    """
    Dibuja los keypoints detectados por ORB o SIFT y guarda la imagen en output/.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, _ = detector.detectAndCompute(gris, None)
    img_kp = cv2.drawKeypoints(
        img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    nombre_archivo = f"{nombre_descriptor}_{nombre_prueba}_keypoints.jpg"
    cv2.imwrite(os.path.join(RUTA_SALIDA, nombre_archivo), img_kp)



def _guardar_imagen_matches(
    img_prueba: np.ndarray,
    img_ref: np.ndarray,
    detector,
    matcher_cv,
    nombre_descriptor: str,
    nombre_prueba: str,
    nombre_clase: str,
) -> None:
    """
    Dibuja las coincidencias entre la imagen de prueba y la referencia ganadora,
    y guarda el resultado en output/.
    """
    gris_prueba = cv2.cvtColor(img_prueba, cv2.COLOR_BGR2GRAY)
    gris_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gris_prueba, None)
    kp2, des2 = detector.detectAndCompute(gris_ref, None)

    if des1 is None or des2 is None:
        return

    pares = matcher_cv.knnMatch(des1, des2, k=2)
    good = []
    for par in pares:
        if len(par) == 2:
            m, n = par
            if m.distance < 0.75 * n.distance:
                good.append([m])

    img_matches = cv2.drawMatchesKnn(
        img_prueba, kp1, img_ref, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    nombre_archivo = f"{nombre_descriptor}_{nombre_prueba}_matches_{nombre_clase}.jpg"
    cv2.imwrite(os.path.join(RUTA_SALIDA, nombre_archivo), img_matches)


def correr_benchmark() -> None:
    """
    Punto de entrada del benchmark.

    1. Carga referencias y pruebas.
    2. Precalcula descriptores de los 3 matchers.
    3. Itera sobre cada imagen de prueba × descriptor.
    4. Escribe results.csv y las imágenes anotadas.
    5. Imprime tabla resumen.
    """
    os.makedirs(RUTA_SALIDA, exist_ok=True)

    # --- Carga de imágenes ---
    try:
        referencias = load_images_from_folder(RUTA_REFERENCIAS)
    except FileNotFoundError as e:
        print(f"[benchmark] Error: {e}")
        sys.exit(1)

    try:
        pruebas = load_images_from_folder(RUTA_PRUEBAS)
    except FileNotFoundError as e:
        print(f"[benchmark] Error: {e}")
        sys.exit(1)

    if not referencias:
        print("[benchmark] Error: no hay imágenes de referencia en data/references/")
        sys.exit(1)

    if not pruebas:
        print("[benchmark] Error: no hay imágenes de prueba en data/tests/")
        sys.exit(1)

    # Redimensionar referencias para consistencia entre descriptores
    referencias_redim = {k: resize_keeping_aspect(v, 640) for k, v in referencias.items()}

    # --- Inicialización de los 3 matchers ---
    orb = ORBMatcher(nfeatures=1000)
    sift = SIFTMatcher()
    akaze = AKAZEMatcher()

    print("[benchmark] Precalculando descriptores...")
    orb.precompute(referencias_redim)
    sift.precompute(referencias_redim)
    akaze.precompute(referencias_redim)

    matchers = [
        ("ORB", orb),
        ("SIFT", sift),
        ("AKAZE", akaze),
    ]

    # --- Evaluación ---
    filas_csv: list[dict] = []
    total = 0
    correctos = 0

    for nombre_prueba, img_prueba in pruebas.items():
        img_redim = resize_keeping_aspect(img_prueba, 640)

        # Nombre de la etiqueta verdadera: se asume que el nombre del archivo
        # de prueba contiene el nombre de la clase (ej. "credencial_test1" → "credencial")
        etiqueta_real = _inferir_etiqueta(nombre_prueba, list(referencias.keys()))

        for nombre_desc, matcher in matchers:
            inicio = time.perf_counter()
            clase_pred, scores = find_id(img_redim, matcher)
            tiempo_ms = (time.perf_counter() - inicio) * 1000

            mejor_score = max(scores) if scores else 0.0
            es_correcto = (clase_pred == etiqueta_real) if etiqueta_real else None

            filas_csv.append({
                "test_image": nombre_prueba,
                "descriptor": nombre_desc,
                "predicted": clase_pred if clase_pred else "—",
                "true_label": etiqueta_real if etiqueta_real else "desconocido",
                "score": f"{mejor_score:.4f}",
                "time_ms": f"{tiempo_ms:.2f}",
                "correct": str(es_correcto),
            })

            total += 1
            if es_correcto:
                correctos += 1

        # --- Imágenes anotadas ---
        _guardar_imagen_keypoints_prueba(img_redim, orb._detector, "ORB", nombre_prueba)
        _guardar_imagen_keypoints_prueba(img_redim, sift._detector, "SIFT", nombre_prueba)
        _guardar_imagen_keypoints_prueba(img_redim, akaze._detector, "AKAZE", nombre_prueba)

        # Guardar matches con la referencia ganadora de ORB, SIFT y AKAZE
        for nombre_desc, matcher, norm in [
            ("ORB", orb, cv2.NORM_HAMMING),
            ("SIFT", sift, cv2.NORM_L2),
            ("AKAZE", akaze, cv2.NORM_HAMMING),
        ]:
            scores = matcher.score_frame(img_redim)
            if scores and max(scores) >= matcher.threshold:
                clase_ganadora = matcher.class_names[scores.index(max(scores))]
                img_ref_ganadora = referencias_redim[clase_ganadora]
                _guardar_imagen_matches(
                    img_redim,
                    img_ref_ganadora,
                    matcher._detector,
                    cv2.BFMatcher(norm),
                    nombre_desc,
                    nombre_prueba,
                    clase_ganadora,
                )

    # --- Escribir CSV ---
    campos = ["test_image", "descriptor", "predicted", "true_label",
              "score", "time_ms", "correct"]
    with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_csv)

    print(f"\n[benchmark] results.csv guardado en: {RUTA_CSV}")
    print(f"[benchmark] Imágenes anotadas guardadas en: {RUTA_SALIDA}/")

    # --- Tabla resumen en consola ---
    _imprimir_tabla(filas_csv)

    if total > 0:
        print(f"\nPrecisión global (cuando etiqueta real es conocida): "
              f"{correctos}/{total} = {100 * correctos / total:.1f}%")


def _guardar_imagen_keypoints_prueba(
    img: np.ndarray, detector, nombre_descriptor: str, nombre_prueba: str
) -> None:
    """Dibuja keypoints del detector sobre img_prueba y guarda en output/."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, _ = detector.detectAndCompute(gris, None)
    if kp:
        img_kp = cv2.drawKeypoints(
            img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    else:
        img_kp = img.copy()
    nombre_archivo = f"{nombre_descriptor}_{nombre_prueba}_keypoints.jpg"
    cv2.imwrite(os.path.join(RUTA_SALIDA, nombre_archivo), img_kp)


def _inferir_etiqueta(nombre_prueba: str, clases: list[str]) -> str | None:
    """
    Infiere la etiqueta verdadera buscando el nombre de una clase dentro del
    nombre del archivo de prueba (comparación insensible a mayúsculas).

    Ejemplo: "credencial_test1" → "credencial" si "credencial" está en clases.

    Args:
        nombre_prueba: Nombre del archivo de prueba (sin extensión).
        clases:        Lista de nombres de clases disponibles.

    Returns:
        Nombre de la clase encontrada, o None si no hay coincidencia.
    """
    nombre_lower = nombre_prueba.lower()
    for clase in clases:
        if clase.lower() in nombre_lower:
            return clase
    return None


def _imprimir_tabla(filas: list[dict]) -> None:
    """Imprime una tabla resumen de resultados en consola."""
    ancho = 78
    print("\n" + "=" * ancho)
    print(f"{'IMAGEN':<22} {'DESCRIPTOR':<8} {'PREDICCIÓN':<16} {'REAL':<16} "
          f"{'SCORE':>8} {'ms':>7}")
    print("-" * ancho)

    for fila in filas:
        print(
            f"{fila['test_image']:<22} {fila['descriptor']:<8} "
            f"{fila['predicted']:<16} {fila['true_label']:<16} "
            f"{fila['score']:>8} {fila['time_ms']:>7}"
        )

    print("=" * ancho)


if __name__ == "__main__":
    correr_benchmark()
