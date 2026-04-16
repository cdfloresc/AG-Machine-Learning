# RESUMEN EJECUTIVO: ALGORITMOS GENÉTICOS EN MACHINE LEARNING
# CURSO: INTELIGENCIA ARTIFICIAL
# INSTITUCIÓN: UNIVERSIDAD NACIONAL DEL ALTIPIANO - PUNO

---

1. OBJETIVO
Comprender y aplicar Algoritmos Genéticos (AG) para optimizar procesos críticos en el aprendizaje de máquina: selección de características, sintonización de hiperparámetros y diseño de arquitecturas neuronales (Neuroevolución).

2. CASOS DE ESTUDIO IMPLEMENTADOS

A. FEATURE SELECTION (SELECCIÓN DE CARACTERÍSTICAS)
Se aplicó un AG para reducir la dimensionalidad del dataset "Breast Cancer".
- Representación: Cromosoma binario (1: incluye característica, 0: excluye).
- Función de Aptitud: Precisión (Accuracy) del modelo DecisionTreeClassifier.
- Fragmento Esencial:
    def fitness_func(instance, solution, solution_idx):
        selected_features = [i for i, bit in enumerate(solution) if bit == 1]
        if len(selected_features) == 0: return 0
        clf.fit(X_train[:, selected_features], y_train)
        return accuracy_score(y_test, clf.predict(X_test[:, selected_features]))

B. HYPERPARAMETER OPTIMIZATION (OPTIMIZACIÓN DE HIPERPARÁMETROS)
Optimización de los parámetros 'C' y 'gamma' para un modelo Support Vector Classifier (SVC) con el dataset "Iris".
- Representación: Cromosoma de valores continuos (genes reales).
- Función de Aptitud: Maximización del acierto en el conjunto de prueba.
- Fragmento Esencial:
    def fitness_func(instance, solution, solution_idx):
        c_param, gamma_param = solution
        svc = SVC(C=c_param, gamma=gamma_param)
        svc.fit(X_train, y_train)
        return accuracy_score(y_test, svc.predict(X_test))

C. NEUROEVOLUTION (NEUROEVOLUCIÓN)
Evolución de la arquitectura de una Red Neuronal (MLPClassifier) para el dataset "Digits".
- Representación: Genes que codifican el número de neuronas, tasa de aprendizaje y máximo de iteraciones.
- Función de Aptitud: Rendimiento de la red en clasificación de imágenes de dígitos.
- Fragmento Esencial:
    def fitness_func(instance, solution, solution_idx):
        h_layer, lr, m_iter = int(solution[0]), solution[1], int(solution[2])
        mlp = MLPClassifier(hidden_layer_sizes=(h_layer,), learning_rate_init=lr, max_iter=m_iter)
        mlp.fit(X_train, y_train)
        return accuracy_score(y_test, mlp.predict(X_test))

3. CICLO DEL ALGORITMO GENÉTICO (PROCESO)
En los tres ejemplos se siguió el ciclo evolutivo estándar mediante la librería PyGAD:
- Inicialización: Se crea una población aleatoria de 10 soluciones posibles.
- Evaluación: Cada solución pasa por la función de aptitud mencionada arriba.
- Selección: Se utiliza "Steady State Selection" para elegir a los mejores padres.
- Cruzamiento (Crossover): Intercambio de genes entre padres (Single Point) para crear hijos.
- Mutación: Alteración aleatoria de genes para mantener diversidad y evitar óptimos locales.
- Terminación: El proceso se repite por un máximo de 50 generaciones.

4. CONCLUSIONES Y RESULTADOS
- La selección de características permitió simplificar el modelo sin perder precisión significativa.
- El AG encontró combinaciones de hiperparámetros de SVC de forma más eficiente que una búsqueda aleatoria.
- La neuroevolución demostró que es posible automatizar el diseño de redes neuronales buscando configuraciones óptimas de capas y aprendizaje.

---
ENLACE GITHUB: [INSERTAR TU LINK AQUÍ]