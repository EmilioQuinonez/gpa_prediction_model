# Sleep vs GPA in College

Regresión lineal múltiple hecha a mano (con descenso de gradiente, sin `scikit-learn`) para predecir el GPA del semestre (`term_gpa`) de estudiantes universitarios a partir de sus datos de sueño, historial académico y variables demográficas. Como validación adicional, se compara contra un `LinearRegression` y un `RandomForestRegressor` de `scikit-learn`.

## Archivos

- `E1_Emilio López_A01711977.py` — script principal: limpieza de datos, EDA, encoding, split train/validate/test, entrenamiento del modelo a mano y comparación con `scikit-learn`.
- `college_sleep_and_gpa.csv` / `study_cohort_reference.csv` — datasets ([fuente en Kaggle](https://www.kaggle.com/datasets/kylefengkfeng209/sleep-vs-gpa-in-college/data)).
- `Reporte_E1_Emilio_Lopez_A01711977.tex` / `.pdf` — reporte del proyecto.
- `*.png` — gráficas generadas por el script (correlación, histogramas, dispersión, curvas de entrenamiento, predicho vs. real del modelo a mano y del framework).

## Cómo correrlo

```bash
pip install pandas numpy matplotlib scikit-learn
python "E1_Emilio López_A01711977.py"
```

El script imprime en consola cada paso del análisis (nulos, duplicados, columnas eliminadas, resultados del modelo) y guarda las gráficas como `.png` en la misma carpeta.

## Resultado

Se comparan dos estrategias para los nulos de `term_units`/`term_load_z` (imputar con la mediana vs. eliminar filas), con un split de 70/15/15 (train/validate/test). La imputación con mediana da mejor resultado y es más estable entre particiones: R² de 0.458/0.445/0.444 contra 0.430/0.276/0.412 de la otra estrategia.

Como validación, un `LinearRegression` de `scikit-learn` entrenado sobre los mismos datos da un R² de test prácticamente idéntico (0.444), confirmando que el descenso de gradiente converge al mismo resultado. Un `RandomForestRegressor` sobreajusta (R² de 0.916 en train contra 0.396 en test) sin superar al modelo lineal, lo que sugiere que la relación con `term_gpa` es, en esencia, lineal. Detalles completos en el reporte.
