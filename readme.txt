# Herramientas para análisis exploratorio de datos en MATLAB

Este repositorio contiene un conjunto de funciones en MATLAB utilizadas para aplicar métodos multivariantes de análisis exploratorio de datos, en concreto **ASCA** y **MEDA**.

##  Archivos incluidos

- `asca.m`: Implementación del método ASCA (ANOVA Simultaneous Component Analysis).
- `meda.m`: Implementación del método MEDA (Missing data methods for Exploratory Data Analysis).
- `loadings.m`: Función auxiliar para obtener las cargas (loadings) de los efectos.
- `parglm.m`: Modelo lineal generalizado para la estimación de parámetros por efecto.
- `pca_pp.m`: Función para aplicar PCA con preprocesamiento.
- `preprocess2D.m`: Preprocesamiento de datos bidimensionales (centrado, escalado, etc.).
- `scores.m`: Cálculo de puntuaciones (scores) a partir de los efectos.

##  Aplicación

Este conjunto de scripts puede utilizarse para:

- Descomponer efectos de factores sobre datos multivariantes
- Analizar estructuras subyacentes en diseños experimentales
- Visualizar puntuaciones y cargas
- Evaluar significancia estadística y visual de los efectos

## Origen

El conjunto de funciones incluidas en este repositorio fue proporcionado por el **Prof. José Camacho Páez** como material docente del curso titulado **"Métodos para el Análisis Exploratorio de Datos MEDA"**, impartido en la Universidad de Granada. 





