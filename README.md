# Sistema de Recomendaciones de Productos Sephora

Este proyecto implementa un sistema de recomendaciones para productos de Sephora utilizando dos enfoques principales: basado en contenido y basado en colaboración. El objetivo es proporcionar recomendaciones personalizadas a los usuarios en función de sus preferencias y comportamientos anteriores.

## Características

- **Recomendación Basada en Contenido**: Utiliza las características del producto (reviews de otros usuarios, categoría del producto, etc.) para recomendar productos similares a aquellos que el usuario ha visto o comprado previamente.
- **Recomendación Basada en Colaboración**: Utiliza las interacciones de los usuarios (compras, calificaciones) para recomendar productos que otros usuarios con gustos similares han disfrutado.
- **Interfaz de Usuario**: Interfaz amigable para mostrar recomendaciones y permitir a los usuarios interactuar con el sistema.

## Tecnologías Utilizadas

- **Lenguaje de Programación**: Python
- **Bibliotecas**: pandas, numpy, seaborn, matplotlib, scikit-learn, nltk, pyspark
- **Framework de Desarrollo Web**: Streamlit

## Proceso de Desarrollo


1. **Limpieza y Análisis Descriptivo de Productos**: Se realiza un análisis exploratorio de los productos de Sephora para entender sus características y atributos relevantes.
   
2. **Análisis de Lenguaje Natural de Reviews**: Se aplican técnicas de procesamiento de lenguaje natural (NLP) a las reseñas de productos para extraer información útil sobre preferencias y opiniones de los usuarios.
   
3. **Reducción de Dimensionalidad con PCA**: Utilizando PCA (Análisis de Componentes Principales), se reduce la dimensionalidad de los datos para poder realizar recomendaciones basadas en contenido de manera eficiente.

4. **Recomendaciones basadas en contenido**: Desarrollamos un sistema de recomendación por similitud que sugiere productos basándose en la vectorización TF-IDF y la similitud del coseno. Esto significa que, al ingresar un producto, el sistema devuelve productos con descripciones de reseñas y características similares, ofreciendo al usuario recomendaciones relevantes y personalizadas.
   
5. **Recomendaciones basadas en colaboración**: Utilizando ALS descompusimos la gran matriz de interacciones entre usuarios y productos en dos matrices más pequeñas, alternando la minimización del error cuadrático medio entre ambas matrices hasta converger. Con esto, un usuario puede ingresar algunos productos y su respectivo rating y recibir recomendaciones. A partir de estas matrices, ALS predice las interacciones futuras calculando productos internos entre los factores de usuarios y productos. Estas predicciones se utilizan para recomendar productos a los usuarios en función de sus preferencias implícitas y explícitas, capturadas a través de sus interacciones anteriores.
