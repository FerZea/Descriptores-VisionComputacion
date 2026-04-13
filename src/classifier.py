"""
classifier.py — Función genérica de clasificación por descriptores.

find_id() recibe un frame y cualquier matcher que implemente la interfaz
(precompute / score_frame / threshold / class_names) y devuelve el nombre
de la clase detectada junto con la lista de scores.

Esta abstracción permite que main.py y benchmark.py sean agnósticos al descriptor.
"""

import numpy as np


def find_id(
    frame: np.ndarray,
    matcher,
) -> tuple[str | None, list[float]]:
    """
    Identifica el objeto en el frame usando el matcher proporcionado.

    Lógica:
      1. Obtiene la lista de scores (uno por clase) del matcher.
      2. Si no hay scores, devuelve None.
      3. Si el score máximo supera matcher.threshold, devuelve el nombre de clase.
      4. De lo contrario, devuelve None (evita falsos positivos).

    Args:
        frame:   Imagen BGR del objeto a identificar.
        matcher: Instancia de ORBMatcher, SIFTMatcher o CannyMatcher
                 (ya inicializada con precompute).

    Returns:
        Tupla (nombre_clase | None, lista_de_scores).
        nombre_clase es None si ningún score supera el threshold.
    """
    scores = matcher.score_frame(frame)

    if not scores:
        return None, scores

    mejor_score = max(scores)

    if mejor_score < matcher.threshold:
        return None, scores

    nombre_clase = matcher.class_names[scores.index(mejor_score)]
    return nombre_clase, scores
