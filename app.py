import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import obses_class

#Cargar metricas

acuracy_svm, recall_svm, cm_svm = obses_class.SVM_C()
acuracy_knn, recall_knn, cm_knn = obses_class.KNN_C()
acuracy_rfc, recall_rfc, cm_rfc = obses_class.RFC_C()

#Matriz confusion

labels = ['Peso Bajo', 'Peso Normal', 'Sobrepeso', 'Obesidad']

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.title('Matriz de confusion')
    st.pyplot(plt) 

# Cargar modelos



# Título de la aplicación
st.markdown('<h1><center>Modelos de Clasificación con Machine Learning</center></h1>',unsafe_allow_html=True)
st.write("<h3><center>Luis Armando García Rodríguez (GARMA)</center><h2>",unsafe_allow_html=True)

st.write("""
         <style>
           p {
            text-align: justify;
             }
         </style>
         <h4>Presentación</h4>
         <p>Los modelos de clasificación de Machine Learning son algoritmos matemáticos y de álgebra lineal, que apoyados del poder computacional, se entrenan para clasificar datos en grupos o clases. Por ejemplo, supongamos que tuvieramos una caja de frutas recolectadas aleatoriamente, y que deseamos ordenarlas por tipo: las manzanas con las manzanas, los plátanos con los plátanos, las guayabas con las guayabas etc. Como humanos podemos hacer esto facilmente, y ahora, gracias al poder computacional, las maquinas tambien pueden hacerlo, para ello, la maquina necesita datos como el largo y ancho de cada fruta, su color, su forma, y la información de otros atributos; de tal manera que la maquina aprenderá las cualidades de cada fruta, y a partir de ellas podrá clasificar cada nueva fruta de la caja de recolección según las caracteristicas de la misma.<p>
         <p>Existen distintos tipos de modelos de clasificación en Machine Learning, como vecinos más cercanos (KNN), bosques aleatorios de decisión de clasificación (RFC), máquinas de soporte vectorial (SVM), entre otros. Cada uno tiene sus pros y contras,  y se adaptan unos mejores que otros según el tipos de problema de clasificación que se presente. La elección del modelo depende de la naturaleza de los datos y del problema específico que se planea resolver. A continuación se presentan algunos de los modelos más comunes, famosos y usados en el Machine Learning.<p>
          """,unsafe_allow_html=True)

st.markdown("""
            <h4>Modelos de Clasificasión</h4>
            <p><strong>SVM (Máquinas de Soporte Vectorial)</strong></p>
            <p>Las Máquinas de Soporte Vectorial (SVM, por sus siglas en inglés) son un modelo de clasificación que busca encontrar el hiperplano que mejor separé las clases de datos en el espacio de características. Como si tuvieramos dos grupos de puntos en un papel, y quisieramos trazar una línea que los separe lo mejor posible; eso lo que hace SVM, solo que puede hacerlo en dimensiones más altas. Lo especial de SVM es que no solo busca cualquier línea que separe los grupos, sino la que deje mayor espacio entre los puntos más cercanos de cada grupo a la línea, a esto se le conoce como el soporte vectorial.</p>
            <p><strong>KNN (K Vecinos más Cercanos)</strong></p>
            <p>K Vecinos más Cercanos (KNN en ingles) es un algoritmo que busca es clasificar un dato no visto basándose en la clasificación de sus vecinos más cercanos. Por ejemplo, si eligimos K=5, el algoritmo buscará los 5 puntos más cercanos al punto que queremos clasificar, y este nuevo punto será asignado a la clase más común entre esos 5 vecinos. Esto lo logra gracias al concepto de distancia, que en el caso mas común, se aplica con la distancia euclidea.<p/>
            <p><strong>RFC (Bosques Aleatorios de Clasificación)</p>
            <p>Los Bosques Aleatorios (RFC, Random Forest Classifier) son  modelos de clasificación que utiliza múltiples árboles de decisión para mejorar la precisión de la clasificación. Un árbol de decisión es un flujo de preguntas que lleva a la respuesta más probable, basándose en las características de los datos. (Como jugar Akinator) Un solo árbol puede ser muy sensible a los datos con los que se entrena, lo que puede llevar a un sopbreajuste y a una clasificación incorrecta con nuevos datos. Para superar esto, los Bosques Aleatorios crean muchos árboles de decisión (un bosque), cada uno entrenado con una muestra aleatoria de los datos, y luego hacen que estos árboles "voten" sobre la clasificación de los nuevos datos. Esto hace que el modelo sea robusto, menos propenso al sobreajuste, y en general, un gran modelo de clasificación.<p>
            <p>En esta aplicación web, he desarrollado dichos modelos de clasificación de ML para poder compararlos entre sí. El objetivo de este proyecto es probarlos y evaluar los mismos y su capacidad y eficencia en la clasificación. Para ello, utilizaremos un caso práctico con datos reales, donde vamos a clasificar el estado del peso de las personas, en función de su altura y peso.<p>
            <p><h4>Caso practico</h4><p>
            <p><strong>El IMC</strong></p>
            <p>Según el ISSSTE, el índice de masa corporal (IMC) sirve para medir la relación entre el peso y la estatura, lo que  permite identificar el sobrepeso y la obesidad en adultos. Para calcularlo se divide el peso de una persona en kilos, entre el cuadrado de estatura en metros. Un IMC alto indica una grasa corporal alta y un IMC bajo indica una grasa corporal demasiado baja. Una vez que se obtiene el cálculo del índice de masa corporal, se puede interpretar la condición en la que se encuentra la persona, de la siguiente forma:<p>
            <ul>
              <li> Menor a 18.49 = peso bajo</li>
              <li>18.50 a 24.99 = peso normal</li>
              <li>25.00 a 29.99 = sobrepeso</li>
              <li> Mayor a 30  = obesidad</li>
            </ul>
            <p> Atravez de los modelos de Machine Learnig podemos hacer una prediccion de la condicion de peso en la que una persona se encuentra, con solo obtener dos datos que es la altura y el peso. Así que elige un modelo, llena el formulario con ambos datos (altura y peso) y obten la predicción de la condicion de peso, así como una visión al rendimiento del modelo de clasificación seleccionado.</p>
            <br>
            """,unsafe_allow_html=True)

    # Formulario para entrada de datos
with st.form(key='mi_formulario'):
    st.markdown("### Elige un modelo, proporciona tu altura, peso y obtén tu predicción!!")
    
    modelo_seleccionado = st.selectbox("Selecciona un modelo", ["SVM", "KNN", "RFC"])
    altura_mts = st.number_input('Altura (en mts)', min_value=0.00, value=1.70, max_value=2.50, step=0.01)
    altura_cm = altura_mts*100
    peso = st.number_input('Peso (en kg)', min_value=10, value=80, max_value=150)
    
    # Botón de envío del formulario
    enviar = st.form_submit_button(label='Estimar con Machine Learning')

if enviar:
    imc = peso / (altura_mts ** 2)
    Observacion = np.array([[altura_cm, peso, imc]])
    
    if modelo_seleccionado == "SVM":
        MODEL = load("SVM.joblib")
        acuracy_model = acuracy_svm
        recall_model = recall_svm
        cm = cm_svm
    elif modelo_seleccionado == "KNN":
        MODEL = load("KNN.joblib")
        acuracy_model = acuracy_knn
        recall_model = recall_knn
        cm = cm_knn
    elif modelo_seleccionado == "RFC":
        MODEL = load("RFC.joblib")
        acuracy_model = acuracy_rfc
        recall_model = recall_rfc
        cm = cm_rfc
    
    prediccion = MODEL.predict(Observacion)
    if prediccion[0] == 1:
        clasif = "peso Bajo"
    elif prediccion[0] == 2:
        clasif = "peso Normal"
    elif prediccion[0] == 3:
        clasif = "sobrepeso"
    elif prediccion[0] == 4:
        clasif = "obesidad"
    st.write("---------------------------------")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"<strong>Evaluación del modelo {modelo_seleccionado}</strong>",unsafe_allow_html=True)
        st.write(f"<p>El modelo desarrollado tienen una accuracy de {acuracy_model} y un recall de {recall_model}, lo que indica una gran calidad del modelo {modelo_seleccionado}. De tal modo que las predicciones son muy confiables. Podemos observar la matriz de confusión, que nos da una visión detallada de como el modelo {modelo_seleccionado} ha acertado y desacertado sus predicciones de clasificación en comparación a las clasificaciones reales.<p>",unsafe_allow_html=True)
        st.write(f"<p><strong>Predicción:</strong> el modelo {modelo_seleccionado} ha clasificado el estado del peso como: <strong>{clasif}</strong>",unsafe_allow_html=True)
        
    
    
    with col2:
        
        plot_confusion_matrix(cm, labels)
    

    st.write("----------------------------------------")
    st.markdown("""<p><h4>Referencias</h4><p/>
                   <ul>
                      <li>ISSTE (s.f.).¿Qué es el índice de masa corporal? gob.mx.https://www.gob.mx/issste/articulos/que-es-el-indice-de-masa-corporal</li>
                      <li>Cristianini, N., & Shawe-Taylor, J. (2000). An Introduction to Support Vector Machines. Cambridge University Press.</li>
                      <li>Altman, N., & Deling, A. E. (2010). K-Nearest Neighbors. Cambridge University Press.</li>
                      <li>Liaw, A., & Wiener, M. (2002). Classification and Regression by Random Forests. R News, 2(3), 18-22.</li>
                   </ul>""",unsafe_allow_html=True)