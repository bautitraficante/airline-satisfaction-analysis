# SkyPulse — Airline Satisfaction Dashboard

Aplicación interactiva en Streamlit para explorar la satisfacción de pasajeros, los servicios, los retrasos y los perfiles de riesgo.

## Instalación dentro del proyecto

1. Copiar `app.py` y `requirements.txt` en la raíz del repositorio.
2. Copiar la carpeta `.streamlit` completa en la raíz.
3. Conservar `CONCLUSIONES.md` como documentación del análisis.
4. Confirmar que el dataset limpio exista en:

   `data/processed/airline_train_clean.csv`

5. Confirmar que el modelo entrenado exista en:

   `models/modelo_satisfaccion_random_forest.joblib`

6. Con el entorno virtual activado, instalar dependencias:

   `pip install -r requirements.txt`

7. Ejecutar:

   `streamlit run app.py`

## Secciones

- Torre de control: indicadores generales y satisfacción por clase.
- Segmentos: tipo de viaje, fidelidad y comparación controlada por clase.
- Experiencia: ranking de servicios relacionados con la satisfacción.
- Puntualidad: impacto de los retrasos de llegada.
- Perfiles de riesgo: segmentos ordenados por porcentaje no satisfecho y muestra mínima configurable.
- Modelo predictivo: métricas, matriz de confusión, importancia de variables y simulador de pasajeros.

## Transparencia

La limpieza, el análisis, las decisiones metodológicas y las conclusiones forman parte del proyecto de aprendizaje de Bautista Traficante. El modelado predictivo fue desarrollado como ejercicio guiado, con asistencia de inteligencia artificial en la estructura del código y la explicación de conceptos. La interfaz interactiva y parte de su implementación también fueron creadas con asistencia de IA.
