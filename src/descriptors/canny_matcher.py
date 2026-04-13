"""
canny_matcher.py — Matcher basado en mapas de bordes Canny + Template Matching.

Canny no produce keypoints ni descriptores locales, sino un mapa de bordes binario.
La comparación se hace con cv2.matchTemplate (TM_CCOEFF_NORMED), que mide la
correlación normalizada entre el mapa de bordes del frame y el de cada referencia.

Elección de métrica: TM_CCOEFF_NORMED en lugar de TM_CCORR_NORMED.
  TM_CCOEFF_NORMED resta la media antes de correlacionar, lo que lo hace menos
  sensible a variaciones globales de brillo y más estable con mapas de bordes
  esparcidos. En las pruebas propias dio resultados más consistentes que CCORR.

Threshold final: 0.4 (correlación normalizada, rango 0–1).
  Justificación: con imágenes de prueba limpias (fondo uniforme), el objeto
  correcto alcanza correlaciones > 0.5; otros objetos quedan por debajo de 0.35.
  Un umbral de 0.4 da margen suficiente sin rechazar detecciones válidas.
"""

import cv2
import numpy as np


# Tamaño fijo al que se redimensionan todos los mapas de bordes para
# que matchTemplate pueda comparar imágenes de distintos tamaños.
_TAMANO_BORDES = (300, 300)


class CannyMatcher:
    """
    Identificador de objetos usando mapas de bordes Canny + Template Matching normalizado.

    Attributes:
        threshold    (float):     Correlación mínima para aceptar una clasificación.
        class_names  (list[str]): Nombres de las clases, en orden de precompute().
        umbral_bajo  (int):       Umbral inferior de Canny (hysteresis).
        umbral_alto  (int):       Umbral superior de Canny (hysteresis).
    """

    # Correlación normalizada mínima para considerar una detección válida.
    # Rango: 0.0 – 1.0. Valor 0.4 validado empíricamente.
    threshold: float = 0.4

    def __init__(self, umbral_bajo: int = 50, umbral_alto: int = 150):
        """
        Args:
            umbral_bajo: Umbral inferior del detector Canny (hysteresis).
            umbral_alto: Umbral superior del detector Canny (hysteresis).
        """
        self.umbral_bajo = umbral_bajo
        self.umbral_alto = umbral_alto
        # Mapas de bordes precalculados de las referencias
        self._ref_bordes: list[np.ndarray] = []
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def precompute(self, references: dict[str, np.ndarray]) -> None:
        """
        Precalcula el mapa de bordes Canny para cada imagen de referencia.

        Se llama una sola vez antes del bucle principal.

        Args:
            references: Diccionario {nombre_clase: imagen_bgr}.
        """
        self.class_names = list(references.keys())
        self._ref_bordes = []

        for nombre, img in references.items():
            bordes = self._extraer_bordes(img)
            self._ref_bordes.append(bordes)

    def score_frame(self, frame: np.ndarray) -> list[float]:
        """
        Puntúa el frame actual contra cada referencia precalculada.

        Obtiene el mapa de bordes del frame y lo compara con cada referencia
        usando matchTemplate. El score es el valor máximo de la correlación.

        Args:
            frame: Frame BGR.

        Returns:
            Lista de scores de correlación (float 0–1) por clase.
        """
        if not self._ref_bordes:
            return []

        bordes_frame = self._extraer_bordes(frame)
        scores: list[float] = []

        for bordes_ref in self._ref_bordes:
            # matchTemplate requiere que el template no sea mayor que la imagen
            # Aquí ambos tienen el mismo tamaño fijo (_TAMANO_BORDES), así que
            # el resultado es una matriz 1×1 con el valor de correlación global.
            resultado = cv2.matchTemplate(
                bordes_frame.astype(np.float32),
                bordes_ref.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )
            # min/max del resultado: nos quedamos con el máximo
            _, max_val, _, _ = cv2.minMaxLoc(resultado)
            scores.append(float(max_val))

        return scores

    # ------------------------------------------------------------------
    # Método privado
    # ------------------------------------------------------------------

    def _extraer_bordes(self, img: np.ndarray) -> np.ndarray:
        """
        Convierte una imagen a gris, la redimensiona y aplica Canny.

        Args:
            img: Imagen BGR de cualquier tamaño.

        Returns:
            Mapa de bordes binario de tamaño _TAMANO_BORDES (uint8, 0 o 255).
        """
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        redim = cv2.resize(gris, _TAMANO_BORDES, interpolation=cv2.INTER_AREA)
        bordes = cv2.Canny(redim, self.umbral_bajo, self.umbral_alto)
        return bordes
