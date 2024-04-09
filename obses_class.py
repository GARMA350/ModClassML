import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from joblib import dump

Datos = pd.read_csv("obesity_data.csv")
Datos.shape

categpesos = {
    "Underweight":1,
    "Normal weight":2,
    "Overweight":3,
    "Obese":4
}


Datos['CatObesidad'] = Datos['CatObesidad'].map(categpesos)

for i in range(len(Datos)):
  Peso_al_cuadrado = (Datos.loc[i,"Altura"]/100) * (Datos.loc[i,"Altura"]/100)
  imc_e = Datos.loc[i,"Peso"] / Peso_al_cuadrado
  Datos.loc[i,"IMC_e"] = imc_e

Obesidad = Datos["CatObesidad"].copy()
Predictoras = Datos[["Altura","Peso","IMC_e"]].copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Predictoras, Obesidad, test_size=0.2, shuffle=True,random_state=32)

from sklearn.svm import SVC

# Instanciar el modelo SVM
svm_model = SVC(kernel='linear')  # Puedes cambiar el kernel según necesites

# Entrenar el modelo con los datos de entrenamiento
svm_model.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)

# Calcular la precisión del modelo
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')

cm_svm = confusion_matrix(y_test, y_pred_svm)  # Asegúrate de que y_pred corresponda al modelo que estás evaluando



def SVM_C():
  svm_accuracy = accuracy_svm
  svm_recall = recall_svm
  svm_cm = cm_svm
  return svm_accuracy, svm_recall, svm_cm

dump(svm_model, 'SVM.joblib')

from sklearn.neighbors import KNeighborsClassifier

# Instanciar el modelo KNN con n_neighbors=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo con los datos de entrenamiento
knn_model.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred_knn = knn_model.predict(X_test)

# Calcular la precisión del modelo
accuracy_knn = accuracy_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')

cm_knn = confusion_matrix(y_test, y_pred_knn)  # Asegúrate de que y_pred corresponda al modelo que estás evaluando


def KNN_C():
  knn_accuracy = accuracy_knn
  knn_recall = recall_knn
  knn_cm = cm_knn
  return knn_accuracy, knn_recall, knn_cm

dump(knn_model, 'KNN.joblib')

# Instanciar el modelo RFC
rfc_model = RandomForestClassifier(n_estimators=10,)

# Entrenar el modelo con los datos de entrenamiento
rfc_model.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred_rfc = rfc_model.predict(X_test)

# Calcular la precisión del modelo
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
recall_rfc = recall_score(y_test, y_pred_rfc, average='weighted')

cm_rfc = confusion_matrix(y_test, y_pred_rfc)  # Asegúrate de que y_pred corresponda al modelo que estás evaluando


def RFC_C():
  rfc_accuracy = accuracy_rfc
  rfc_recall = recall_rfc
  rfc_cm = cm_rfc
  return rfc_accuracy, rfc_recall, rfc_cm

dump(rfc_model, 'RFC.joblib')