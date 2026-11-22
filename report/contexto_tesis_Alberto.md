# Contexto de tesis: Alberto Martinez Hernandez

La versión en inglés es la que aparece en la sección I y la conclusión del informe, en la slide 2 y la última, y en el resumen.

## Versión en español

**Contexto.** Mi tesis estudia la cuantificación y estandarización del dolor a partir de EEG. El estándar clínico sigue siendo el autorreporte (por ejemplo, una escala numérica de 0 a 10). Es subjetivo, depende de que la persona pueda comunicarse y no es comparable entre personas: un "6" no significa lo mismo para dos sujetos. Un marcador neural del dolor permitiría una medida más objetiva. Sin embargo, la actividad cortical asociada al dolor varía mucho entre personas, así que un modelo útil tiene que funcionar en sujetos que no vio al entrenarse.

**Por qué este proyecto.** El problema metodológico es el mismo que en mi tesis. Hay que inferir un estado subjetivo, definido de forma relativa a cada persona, a partir de un EEG muy variable entre sujetos. Y hay que medir si la representación de la señal (cruda o tiempo-frecuencia) cambia la generalización a sujetos nuevos. La etiqueta de "respuesta lenta", relativa a la sesión de cada sujeto, es análoga a una calificación de dolor normalizada por persona. Mi base de datos de dolor aún no está completa (la meta es al menos 40 sujetos), así que el conjunto público de conducción sirve de banco de pruebas para el pipeline. El vínculo con mi tesis es indirecto, porque este conjunto no mide dolor.

**Siguiente paso.** Cuando mi base de datos llegue a 40 sujetos, aplicaré el mismo protocolo para predecir la intensidad de dolor relativa a cada sujeto:
- validación cruzada por sujeto;
- normalización por sesión, declarada como calibración sin etiquetas;
- EEGNet sobre señal cruda frente a ResNet18 sobre escalogramas;
- bootstrap por sujeto.

Dado el resultado aquí, el punto de partida será EEGNet sobre la señal cruda. Antes de atribuir el desempeño a actividad cortical, repetiré el diagnóstico por canal y por banda y añadiré control de artefactos oculares y musculares (ICA/EOG). En dolor, ese riesgo es mayor que en conducción.

## English version (as in the report)

Alberto's thesis studies the quantification and standardization of pain from EEG. The clinical standard is still self-report (e.g., a 0–10 numerical rating scale). It is subjective, requires the person to communicate, and is not comparable across people. An EEG marker of pain could give a more objective measure, but pain-related cortical activity varies strongly between individuals, so a useful model must work on people it has not seen. His pain dataset is still being collected (target: at least 40 subjects).

Next step (Alberto): once his pain dataset reaches 40 subjects, he will apply the same protocol to predict subject-relative pain intensity: subject-wise cross-validation, per-session normalization declared as unlabeled calibration, raw-signal EEGNet vs. scalogram ResNet18, and a subject bootstrap. Given the result here, he will start from EEGNet on the raw signal. He will also repeat the channel/band diagnostic to rule out ocular and muscle artifacts, which are a larger risk in pain paradigms than in driving.
