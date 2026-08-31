# Sleep vs GPA in College

Regresión lineal múltiple hecha a mano (con descenso de gradiente, sin framework) para predecir el GPA del semestre (`term_gpa`) de estudiantes universitarios a partir de sus datos de sueño, historial académico y variables demográficas.

## Archivos

- `E1_Emilio López_A01711977.py` — script principal: limpieza de datos, EDA, encoding, split train/test y entrenamiento del modelo.
- `college_sleep_and_gpa.csv` / `study_cohort_reference.csv` — datasets ([fuente en Kaggle](https://www.kaggle.com/datasets/kylefengkfeng209/sleep-vs-gpa-in-college/data)).
- `Reporte_E1_Emilio_Lopez_A01711977.tex` / `.pdf` — reporte del proyecto.
- `*.png` — gráficas generadas por el script (correlación, histogramas, dispersión, curvas de entrenamiento, predicho vs. real).

## Cómo correrlo

```bash
pip install pandas numpy matplotlib
python "E1_Emilio López_A01711977.py"
```

El script imprime en consola cada paso del análisis (nulos, duplicados, columnas eliminadas, resultados del modelo) y guarda las gráficas como `.png` en la misma carpeta.

## Resultado

Se comparan dos estrategias para los nulos de `term_units`/`term_load_z` (imputar con la mediana vs. eliminar filas). La imputación con mediana da mejor resultado: R² de 0.424 en test contra 0.392 de la otra estrategia. Detalles completos en el reporte.
