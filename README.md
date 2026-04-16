# Resumen: Algoritmos Genéticos
 
**Actividad:** 02

---

## 1. Introducción
El presente documento detalla la aplicación de **Algoritmos Genéticos (AG)** para optimizar tres pilares del aprendizaje de máquina. Se utiliza la librería `PyGAD` para simular el proceso evolutivo (selección, cruce y mutación) con el fin de encontrar soluciones óptimas en espacios de búsqueda complejos.

## 2. Desarrollo de Casos de Estudio

### A. Feature Selection (Selección de Características)
Se implementó un AG para identificar las variables más influyentes en el rendimiento de un clasificador, eliminando datos redundantes.
* **Modelo Base:** `DecisionTreeClassifier`.
* **Representación:** Cromosoma binario (1 = característica activa, 0 = inactiva).
* **Función de Aptitud:**
    ```python
    def fitness_func(instance, solution, solution_idx):
        selected_features = [i for i, bit in enumerate(solution) if bit == 1]
        if len(selected_features) == 0: return 0
        clf.fit(X_train[:, selected_features], y_train)
        return accuracy_score(y_test, clf.predict(X_test[:, selected_features]))
    ```

### B. Hyperparameter Optimization
Se optimizaron los parámetros críticos de una **Máquina de Soporte de Vectores (SVM)** para maximizar su precisión en el dataset Iris.
* **Parámetros:** `C` (Regularización) y `gamma` (Coeficiente del kernel).
* **Enfoque:** Búsqueda heurística que supera en eficiencia a la búsqueda por grilla tradicional.
* **Función de Aptitud:**
    ```python
    def fitness_func(instance, solution, solution_idx):
        c_param, gamma_param = solution
        svc = SVC(C=c_param, gamma=gamma_param)
        svc.fit(X_train, y_train)
        return accuracy_score(y_test, svc.predict(X_test))
    ```

### C. Neuroevolution (Neuroevolución)
Evolución de los parámetros estructurales y de entrenamiento de una **Red Neuronal Multicapa (MLP)**.
* **Genes:** Número de neuronas en capas ocultas, tasa de aprendizaje inicial y máximo de iteraciones.
* **Dataset:** Digits (Reconocimiento de dígitos manuscritos).
* **Función de Aptitud:**
    ```python
    def fitness_func(instance, solution, solution_idx):
        h_layer, lr, m_iter = int(solution[0]), solution[1], int(solution[2])
        mlp = MLPClassifier(hidden_layer_sizes=(h_layer,), learning_rate_init=lr, max_iter=m_iter)
        mlp.fit(X_train, y_train)
        return accuracy_score(y_test, mlp.predict(X_test))
    ```

## 3. Configuración del Algoritmo Genético
Para todos los experimentos se definieron los siguientes hiperparámetros evolutivos:
* **Población inicial:** 10 soluciones.
* **Selección de padres:** `Steady State Selection (SSS)`.
* **Crossover:** Un solo punto (`single_point`).
* **Mutación:** Aleatoria (`random`) para preservar la diversidad genética.
* **Generaciones:** 20 a 50 iteraciones según la complejidad del caso.

## 4. Conclusiones
1.  **Eficiencia:** Los AG logran reducir la dimensionalidad de los datos sin sacrificar significativamente la precisión.
2.  **Adaptabilidad:** La neuroevolución permite encontrar arquitecturas de red que no son evidentes mediante el diseño manual.
3.  **Optimización:** El uso de `PyGAD` facilita la integración de lógica evolutiva en flujos de trabajo estándar de Scikit-Learn.

---
**Enlace al Repositorio:** [Inserta aquí tu link de GitHub]
