# Contexto de tesis: Juan Pablo Acevedo Robles

La versión en inglés es la que aparece en la sección I y la conclusión del informe, en la slide 2 y la última, y en el resumen.

## Versión en español

**Contexto.** Mi tesis estudia la medición de la confianza en sistemas autónomos en tiempo real, a partir de señales fisiológicas multimodales (entre ellas EEG), con la conducción autónoma como banco de pruebas. Hoy la confianza se mide sobre todo con cuestionarios al final de una tarea. Son subjetivos, no permiten seguir cómo cambia la confianza momento a momento y no son comparables entre personas. Una medida fisiológica permitiría seguirla de forma continua y sin interrumpir la tarea. Sin embargo, las señales fisiológicas varían mucho entre personas, así que un modelo útil tiene que funcionar en sujetos que no vio al entrenarse.

**Por qué este proyecto.** El problema metodológico es el mismo que en mi tesis. Hay que inferir un estado subjetivo, definido de forma relativa a cada persona, a partir de un EEG muy variable entre sujetos, y en un contexto de conducción. Y hay que medir si la representación de la señal (cruda o tiempo-frecuencia) cambia la generalización a sujetos nuevos. La etiqueta de "respuesta lenta", relativa a la sesión de cada sujeto, es análoga a un nivel de confianza normalizado por persona. Mi base de datos, grabada en el simulador CARLA, aún no está completa (la meta es al menos 40 sujetos), así que el conjunto público de conducción sirve de banco de pruebas para el pipeline. El vínculo con mi tesis es indirecto, porque este conjunto no mide confianza.

**Siguiente paso.** Cuando mi base de datos llegue a 40 sujetos, aplicaré el mismo protocolo para inferir la confianza relativa a cada sujeto:
- validación cruzada por sujeto;
- normalización por sesión, declarada como calibración sin etiquetas;
- EEGNet sobre señal cruda frente a ResNet18 sobre escalogramas;
- bootstrap por sujeto.

Dado el resultado aquí, el punto de partida será EEGNet sobre la señal cruda. Antes de atribuir el desempeño a actividad cortical, repetiré el diagnóstico por canal y por banda y añadiré control de artefactos oculares, usando los datos de movimiento ocular cuando estén disponibles. En conducción, los movimientos de ojos y cabeza son frecuentes, así que ese riesgo es real.

## English version (as in the report)

Juan Pablo's thesis studies real-time measurement of trust in autonomous systems from multimodal physiological signals, including EEG, with autonomous driving as the testbed. Trust is still measured mostly with questionnaires after a task, which are subjective, cannot follow how trust changes moment to moment, and are not comparable across people. A physiological measure could track it continuously without interrupting the task, but physiological signals vary strongly between individuals, so a useful model must work on people it has not seen. His driving dataset, recorded in the CARLA simulator, is still being collected (target: at least 40 subjects).

Next step (Juan Pablo): once his dataset reaches 40 subjects, he will apply the same protocol to infer subject-relative trust: subject-wise cross-validation, per-session normalization declared as unlabeled calibration, raw-signal EEGNet vs. scalogram ResNet18, and a subject bootstrap. Given the result here, he will start from EEGNet on the raw signal. He will also repeat the channel/band diagnostic and add ocular-artifact control, using eye-movement data where available, since eye and head movements are frequent while driving.
