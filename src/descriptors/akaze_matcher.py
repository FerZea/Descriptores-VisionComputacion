"""
akaze_matcher.py — Matcher basado en el descriptor AKAZE (Accelerated-KAZE).

AKAZE es un descriptor local de keypoints no lineal incluido en OpenCV base
(no requiere opencv-contrib). Combina la detección de keypoints en espacio de
escala no lineal (más estable que ORB ante ruido) con descriptores binarios
M-LDB (Modified Local Difference Binary) de longitud variable.

Por qué NORM_HAMMING:
  AKAZE usa descriptores binarios M-LDB por defecto (cv2.AKAZE_DESCRIPTOR_MLDB),
  por lo que la distancia correcta para compararlos es Hamming, igual que ORB.
  Usar NORM_L2 sería incorrecto y daría resultados degradados.

Pipeline:
  1. cv2.AKAZE_create() detecta keypoints y calcula descriptores M-LDB.
  2. BFMatcher(NORM_HAMMING) + knnMatch(k=2) encuentra los 2 mejores vecinos.
  3. Ratio Test de Lowe (0.75) filtra matches ambiguos.
  4. El score es el conteo de good matches; si supera threshold → detectado.

Threshold final: 20 good matches.
  Justificación: AKAZE genera entre ORB y SIFT en cantidad de keypoints.
  En las pruebas con las 3 clases del proyecto (credencial, libro, videojuego),
  un threshold de 20 es suficiente para distinguir todas las clases sin
  falsos positivos, ya que los objetos correctos alcanzan >100 good matches
  y los incorrectos quedan por debajo de 15.
"""

import cv2
import numpy as np


class AKAZEMatcher:
    """
    Identificador de objetos usando AKAZE + Brute-Force Hamming + Ratio Test de Lowe.

    Attributes:
        threshold   (int):        Mínimo de good matches para aceptar una clasificación.
        class_names (list[str]):  Nombres de las clases, en el orden de precompute().
    """

    threshold: int = 20

    def __init__(self):
        self._detector = cv2.AKAZE_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_descriptors: list[np.ndarray | None] = []
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def precompute(self, references: dict[str, np.ndarray]) -> None:
        """
        Precalcula los descriptores AKAZE para cada imagen de referencia.

        Se llama una sola vez antes del bucle principal para no repetir
        el cálculo en cada frame.

        Args:
            references: Diccionario {nombre_clase: imagen_bgr}.
        """
        self.class_names = list(references.keys())
        self._ref_descriptors = []

        for nombre, img in references.items():
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptores = self._detector.detectAndCompute(gris, None)

            if descriptores is None:
                print(f"[AKAZE] Advertencia: sin descriptores para '{nombre}'.")

            self._ref_descriptors.append(descriptores)

    def score_frame(self, frame: np.ndarray) -> list[float]:
        """
        Puntúa el frame actual contra cada referencia precalculada.

        Aplica Ratio Test de Lowe (umbral 0.75) para filtrar matches ambiguos.
        Maneja el caso en que knnMatch devuelva pares con menos de 2 vecinos.

        Args:
            frame: Frame BGR capturado de cámara o cargado de disco.

        Returns:
            Lista de conteos de good matches, uno por clase (mismo orden que class_names).
        """
        if not self._ref_descriptors:
            return []

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des_frame = self._detector.detectAndCompute(gris, None)

        scores: list[float] = []

        for des_ref in self._ref_descriptors:
            if des_ref is None or des_frame is None:
                scores.append(0.0)
                continue

            pares = self._matcher.knnMatch(des_frame, des_ref, k=2)

            good_matches = 0
            for par in pares:
                if len(par) == 2:
                    m, n = par
                    if m.distance < 0.75 * n.distance:
                        good_matches += 1

            scores.append(float(good_matches))

        return scores
