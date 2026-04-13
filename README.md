# Identificación de objetos con descriptores OpenCV

Segundo examen parcial de Visión Computacional — UASLP, Facultad de Ingeniería.

## Descripción

Programa en Python que identifica al menos 3 objetos usando tres descriptores distintos de OpenCV:

| Descriptor | Método de comparación | Threshold |
|------------|----------------------|-----------|
| **ORB** | BFMatcher Hamming + Ratio Test (0.75) | 15 good matches |
| **SIFT** | BFMatcher L2 + Ratio Test (0.75) | 30 good matches |
| **Canny** | matchTemplate TM_CCOEFF_NORMED | 0.4 (correlación) |

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura de datos

Agrega las imágenes de referencia en `data/references/` con el nombre de la clase:

```
data/references/
├── credencial.jpg
├── libro.jpg
└── caja.jpg

data/tests/
├── credencial_test1.jpg
├── libro_test1.jpg
└── caja_test1.jpg
```

> El nombre del archivo de referencia **es** el nombre de la clase.

## Uso

```bash
# Modo imagen estática
python src/main.py --mode image --descriptor orb --input data/tests/credencial_test1.jpg
python src/main.py --mode image --descriptor sift --input data/tests/libro_test1.jpg
python src/main.py --mode image --descriptor canny --input data/tests/caja_test1.jpg

# Modo webcam en tiempo real
python src/main.py --mode webcam --descriptor orb
python src/main.py --mode webcam --descriptor sift
python src/main.py --mode webcam --descriptor canny

# Benchmark completo (genera results.csv y output/)
python src/main.py --mode benchmark
```

## Salidas del benchmark

- `results.csv` — tabla con predicción, etiqueta real, score y tiempo por imagen/descriptor
- `output/` — imágenes anotadas con keypoints, matches y mapas de bordes

## Referencia

Murtaza's Workshop (2020). *Feature Detection and Matching + Image Classifier Project | OPENCV PYTHON*. YouTube.
