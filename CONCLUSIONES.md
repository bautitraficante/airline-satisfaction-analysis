# Conclusiones del análisis

El análisis de 103.594 respuestas muestra una situación general mejorable: el 43,34 % de los pasajeros se declaró satisfecho, mientras que el 56,66 % quedó neutral o insatisfecho.

La clase de viaje presenta una diferencia marcada. Business alcanzó un 69,43 % de satisfacción, frente al 18,61 % de Eco y al 24,58 % de Eco Plus. Sin embargo, la clase no explica por sí sola el resultado. Entre quienes realizaron viajes personales, la satisfacción fue baja en las tres clases; en cambio, los viajes de negocios en Business registraron el mejor desempeño. Esto indica que la experiencia depende de la combinación entre motivo del viaje, clase y tipo de cliente.

Los clientes fieles tuvieron 47,73 % de satisfacción, 24,05 puntos porcentuales más que los no fieles. Aun así, menos de la mitad de los clientes fieles quedó satisfecha, por lo que la fidelidad no elimina los problemas de experiencia.

Entre los servicios evaluados, el embarque online mostró la relación más fuerte con la satisfacción (correlación de 0,50), seguido por el entretenimiento a bordo (0,40), la comodidad del asiento (0,35) y el servicio a bordo (0,32). Estas asociaciones permiten ordenar prioridades, aunque no demuestran causalidad.

Los retrasos también afectan la experiencia. La satisfacción fue de 47,28 % cuando no hubo demora y cayó a 41,30 % con retrasos de hasta 15 minutos. A partir de los 16 minutos descendió a aproximadamente 35 %. Por eso, el objetivo operativo debería ser evitar demoras y, cuando no sea posible, mantenerlas por debajo de los 15 minutos.

La recomendación final es trabajar en dos frentes: reducir las fricciones digitales y operativas —especialmente embarque online, check-in, información y puntualidad— y diseñar acciones específicas para los segmentos de mayor volumen y menor satisfacción. Antes de invertir, las mejoras deberían validarse mediante pruebas controladas, porque una correlación no permite afirmar que un servicio sea por sí solo la causa de la satisfacción.

## Modelo predictivo

Se comparó una regresión logística con dos versiones de Random Forest. El Random Forest original fue seleccionado como modelo definitivo: alcanzó un 96,26 % de accuracy en el conjunto de prueba y un promedio de 96,10 % en cinco divisiones de validación cruzada.

El modelo identificó correctamente el 97,82 % de los pasajeros realmente neutrales o insatisfechos. Sobre 20.719 casos de prueba, cometió 774 errores: 256 pasajeros insatisfechos fueron clasificados como satisfechos y 518 satisfechos fueron clasificados como insatisfechos.

Las variables más utilizadas por el modelo fueron el embarque online, el Wi-Fi a bordo, el tipo de viaje, la clase y el entretenimiento durante el vuelo. Estas importancias no muestran dirección ni causalidad; deben interpretarse junto con el análisis descriptivo.

## Nota metodológica

El análisis se basa en asociaciones descriptivas del dataset y no en relaciones causales. Los 310 registros sin demora de llegada fueron eliminados durante la limpieza, equivalentes al 0,30 % del conjunto original.

## Transparencia

La limpieza, el análisis, las decisiones metodológicas y las conclusiones fueron trabajadas por Bautista Traficante como proyecto de aprendizaje. El modelado predictivo fue desarrollado como ejercicio guiado, con asistencia de inteligencia artificial en la estructura del código y la explicación de conceptos. La interfaz interactiva y parte de su implementación también fueron creadas con asistencia de IA.
