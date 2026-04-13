"""
sift_matcher.py — Matcher basado en el descriptor SIFT (Scale-Invariant Feature Transform).

SIFT produce descriptores de 128 floats que son invariantes a escala y rotación.
Usa la distancia L2 (Euclidiana) para la comparación, a diferencia de ORB que usa Hamming.
Pipeline análogo al de ORB (video de Murtaza's Workshop, 2020).

Threshold final: 30 good matches.
  Justificación: SIFT genera más keypoints (y más consistentes) que ORB porque
  sus detectores son más estables bajo cambios de escala. Un threshold mayor
  reduce falsos positivos sin sacrificar sensibilidad en las pruebas realizadas.
"""

import cv2
import numpy as np


class SIFTMatcher:
    """
    Identificador de objetos usando SIFT + Brute-Force L2 + Ratio Test de Lowe.

    Attributes:
        threshold   (int):        Mínimo de good matches para aceptar clasificación.
        class_names (list[str]):  Nombres de las clases, en orden de precompute().
    """

    # Umbral de good matches (Lowe ratio test 0.75).
    # SIFT genera más matches buenos que ORB; 30 evita falsos positivos.
    threshold: int = 30

    def __init__(self):
        # cv2.SIFT_create() está disponible en OpenCV >= 4.5 sin contrib
        self._detector = cv2.SIFT_create()
        # NORM_L2 porque los descriptores de SIFT son vectores reales, no binarios
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._ref_descriptors: list[np.ndarray | None] = []
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def precompute(self, references: dict[str, np.ndarray]) -> None:
        """
        Precalcula los descriptores SIFT para cada imagen de referencia.

        Se llama una sola vez antes del bucle principal.

        Args:
            references: Diccionario {nombre_clase: imagen_bgr}.
        """
        self.class_names = list(references.keys())
        self._ref_descriptors = []

        for nombre, img in references.items():
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptores = self._detector.detectAndCompute(gris, None)

            if descriptores is None:
                print(f"[SIFT] Advertencia: sin descriptores para '{nombre}'.")

            self._ref_descriptors.append(descriptores)

    def score_frame(self, frame: np.ndarray) -> list[float]:
        """
        Puntúa el frame actual contra cada referencia precalculada.

        Aplica Ratio Test de Lowe (umbral 0.75) igual que en ORB.
        Maneja pares incompletos de knnMatch.

        Args:
            frame: Frame BGR.

        Returns:
            Lista de conteos de good matches por clase.
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
