"""
orb_matcher.py — Matcher basado en el descriptor ORB (Oriented FAST and Rotated BRIEF).

ORB es libre de patentes y computacionalmente muy eficiente.
Genera descriptores binarios de 32 bytes comparables con la distancia Hamming.
Pipeline basado en el video de Murtaza's Workshop (2020).

Threshold final: 15 good matches.
  Justificación: valor del video de referencia, validado empíricamente con las
  imágenes de prueba. ORB genera menos matches que SIFT por ser más restrictivo
  con los keypoints, por lo que 15 es suficiente para distinguir clases sin
  generar falsos positivos.
"""

import cv2
import numpy as np


class ORBMatcher:
    """
    Identificador de objetos usando el descriptor ORB + Brute-Force + Ratio Test de Lowe.

    Attributes:
        threshold   (int):        Mínimo de good matches para aceptar una clasificación.
        class_names (list[str]):  Nombres de las clases, en el orden de precompute().
    """

    # Umbral de good matches (Lowe ratio test 0.75).
    # Valor 15 tomado del video de referencia y confirmado con pruebas propias.
    threshold: int = 15

    def __init__(self, nfeatures: int = 1000):
        """
        Args:
            nfeatures: Número máximo de keypoints que extrae ORB por imagen.
        """
        self._detector = cv2.ORB_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Descriptores precalculados de las referencias: lista de np.ndarray
        self._ref_descriptors: list[np.ndarray | None] = []
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def precompute(self, references: dict[str, np.ndarray]) -> None:
        """
        Precalcula los descriptores ORB para cada imagen de referencia.

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
                print(f"[ORB] Advertencia: sin descriptores para '{nombre}'.")

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
            # Sin descriptores en la referencia o en el frame → score 0
            if des_ref is None or des_frame is None:
                scores.append(0.0)
                continue

            # kNN con k=2 para aplicar el Ratio Test de Lowe
            pares = self._matcher.knnMatch(des_frame, des_ref, k=2)

            good_matches = 0
            for par in pares:
                # Ignorar pares incompletos (menos de 2 vecinos)
                if len(par) == 2:
                    m, n = par
                    if m.distance < 0.75 * n.distance:
                        good_matches += 1

            scores.append(float(good_matches))

        return scores
