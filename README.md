# Airline Passenger Satisfaction Analysis

Proyecto de análisis de datos y machine learning orientado a comprender y predecir la satisfacción de pasajeros de una aerolínea.

## Objetivo

El proyecto busca responder tres preguntas principales:

1. ¿Qué perfiles de pasajeros presentan menor satisfacción?
2. ¿Qué servicios y factores operativos se relacionan más con la experiencia?
3. ¿Es posible predecir la satisfacción de un pasajero a partir de la información disponible después del vuelo?

## Resultados principales

- 43,34 % de los pasajeros se declaró satisfecho.
- Business alcanzó 69,43 % de satisfacción, frente a 18,61 % en Eco.
- Los clientes fieles presentaron 24,05 puntos porcentuales más de satisfacción que los no fieles.
- El embarque online fue el servicio con mayor correlación con la satisfacción.
- A partir de 16 minutos de retraso, la satisfacción descendió hasta aproximadamente 35 %.
- El Random Forest final alcanzó 96,26 % de accuracy en prueba y 96,10 % promedio en validación cruzada.
- El modelo detectó correctamente el 97,82 % de los pasajeros neutrales o insatisfechos.

## Metodología

1. Exploración y limpieza de datos.
2. Análisis descriptivo y segmentación.
3. Estudio de servicios, puntualidad y perfiles de riesgo.
4. Preparación de variables numéricas y categóricas mediante un pipeline.
5. Comparación entre regresión logística y Random Forest.
6. Control de sobreajuste y validación cruzada.
7. Selección y exportación del modelo final.
8. Desarrollo de un dashboard interactivo en Streamlit.

## Estructura esperada

```text
airline-satisfaction-analysis/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
│       └── airline_train_clean.csv
├── models/
│   └── modelo_satisfaccion_random_forest.joblib
├── notebooks/
│   ├── 01_exploracion_y_limpieza.ipynb
│   ├── 02_analisis_exploratorio.ipynb
│   └── 03_modelo_predictivo.ipynb
├── .streamlit/
│   └── config.toml
├── CONCLUSIONES.md
└── requirements.txt
```

## Ejecutar la aplicación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tecnologías

- Python
- Pandas y NumPy
- Matplotlib y Seaborn
- Plotly y Streamlit
- scikit-learn
- joblib

## Transparencia

La limpieza, el análisis, las decisiones metodológicas y las conclusiones fueron trabajadas por Bautista Traficante como proyecto de aprendizaje.

El modelado predictivo fue desarrollado como ejercicio guiado, con asistencia de inteligencia artificial en la estructura del código y la explicación de conceptos. La interfaz interactiva y parte de su implementación también fueron creadas con asistencia de IA.

Los resultados describen asociaciones presentes en el dataset y no deben interpretarse como relaciones causales.
