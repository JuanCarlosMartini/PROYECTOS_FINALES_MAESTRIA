#!/usr/bin/env python
# coding: utf-8

# <center> <h1> Análisis de Minería de Datos: Adult_Income
# </h1> </center>

# ## Integrantes
# * Edwin Gustavo Lima Dubón 2457001960101
# * César Geovanni López García 2669080550101
# * Ivan Alexis Orozco Fuentes 1683584761202
# * Juan Carlos Martini Palma 2963767440101
# * Josué Ricardo Slansky Rabanales Gómez 2338721400101           

# ## 1. Listado de cada uno de los atributos del dataset, conteniendo el tipo de dato, descripción y el análisis que obtendría de cada uno de ellos.
# 
# <table>
# <thead>
# <tl>
# <th style="text-align: center">Atributo</th>
# <th style="text-align: center">Tipo de dato</th>
# <th style="text-align: center">Descripción</th>
# <th style="text-align: center">Análisis</th>
# </tl>
# </thead>
#     
# <tbody>
# <tl>
# <td style="text-align: center">age</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">La edad de cada individuo </td>
# <td style="text-align: left">Se verificaría la edad en la que oscilan las personas que obteniene un ingreso mayor a los \$50,000 </td>
# </tl>
# 
# <tr>
# <td style="text-align: center">workclass</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">El estado de empleo de cada individuo</td>
# <td style="text-align: left">Se verificaría que tipos de empleados alcanzan la meta de los \$50,000 </td>
# </tr>
#     
# <tr>
# <td style="text-align: center">fnlwgt</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">Número de personas representadas por el censo</td>
# <td style="text-align: left">Determinaría la cantidad de registros utilizados para el análisis estadístico.</td>
# </tr>
#     
# <tr>
# <td style="text-align: center">education</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">El nivel más alto de educación alcanzado por cada individuo.</td>
# <td style="text-align: left">Ayudaría a verificar si el nivel académico influencia en el alzancar la meta de los \$50,000.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">education</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">El nivel más alto de educación alcanzado por cada individuo.</td>
# <td style="text-align: left">Ayudaría a verificar si el nivel académico influencia en el alzancar la meta de los \$50,000.</td>
# </tr>
#     
# <tr>
# <td style="text-align: center">education.num</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">El nivel más alto de educación alcanzado por cada individuo representado de forma numérica.</td>
# <td style="text-align: left">Haría énfasis en el grado puntual en el que se puede o no alcanzar el ingreso esperado.</td>
# </tr>
#     
# <tr>
# <td style="text-align: center">marital.status</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">El estado civil de cada persona.</td>
# <td style="text-align: left">Mostraría si estar en un relación matrial influye o no en alcanzar los \$50,000 anuales. </td>
# </tr>
# 
# <tr>
# <td style="text-align: center">occupation</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">Representa la ocupación de cada individuo.</td>
# <td style="text-align: left">La ocupación que tenga cada persona influirá directamente con el ingreso anual.</td>
# </tr>   
# 
# <tr>
# <td style="text-align: center">relationship</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">Representa la relación que el individuo tiene con otras personas.</td>
# <td style="text-align: left">Este atributo es redundante con el atributo de estado civil, por lo tanto se considera de aporte nulo al análisis.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">race</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">Describe la raza de cada individuo.</td>
# <td style="text-align: left">Se verificaría si la raza influye en los ingresos obtenidos por cada persona.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">sex</td>
# <td style="text-align: left">Numérico, discreto</td>
# <td style="text-align: left">Representa el sexo biológico de cada individuo. 0 represneta a hombre y 1 a mujer</td>
# <td style="text-align: left">Mostraría si exsite una brecha salarial y de ingresos por el sexo que tiene cada individuo.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">capital.gain</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">Ganacias de capital de cada individuo.</td>
# <td style="text-align: left">Se verificaría que el aumento en el capital influye en acumular los \$50,000 anuales.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">capital.loss</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">Pérdida de capital de cada individuo.</td>
# <td style="text-align: left">Se verificaría que la disminución en el capital evtia que se acumule la meta de \$50,000.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">hourse.per.week</td>
# <td style="text-align: left">Numérico, continuo</td>
# <td style="text-align: left">Horas de trabajo por semana reportadas por cada individuo.</td>
# <td style="text-align: left">Se demostraría si trabajar más horas por semana aumenta las posibilidades de estar dentro del grupo de personas que pueden tener \$50,000 o no.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">native.country</td>
# <td style="text-align: left">Categórico</td>
# <td style="text-align: left">País de origen de cada individuo.</td>
# <td style="text-align: left">Se evidenciaría si la pertenencía a un país mejora las posibilidades de obtener el ingreso esperado o no.</td>
# </tr>
# 
# <tr>
# <td style="text-align: center">income</td>
# <td style="text-align: left">Categórico, binario</td>
# <td style="text-align: left">Etiqueta que posee cada individuo al tener una ganacia por año menor o mayor igual a \$50,000.</td>
# <td style="text-align: left">Esta es la variable objetivo, la cual está en función de las demás variables descritas.</td>
# </tr>
# </tbody>
# </table>

# ## 2. Explicacion del problema a resolver
# En Estados Unidos existe una gran desigualdad de la riqueza, lo cual causa mucha preocupación ya que el 1% de la población posee el 39% de las riquezas según información de la reserva federal de Estados Unidos al año 2020. La idea de la igualdad de la riqueza busca garantizar el desarrollo y mejorar la economía del país. 
# Este no es un problema que se presenta en Estados Unidos, si no que se presenta alrededor de las naciones en el mundo las cuales han estado luchando contra la pobreza con el fin de generar una mejor calidad de vida para sus ciudadanos.
# Es por eso por lo que en el presente estudio se busca por medio de técnicas de minería de datos y algoritmos de machine learning proveer una opción para determinar la desigualdad de ingresos en los ciudadanos norte americanos. El data set que se utilizara en este estudio adult_income contiene información sobre los ingresos económicos de los adultos en estados unidos. Estos datos contienen variables categóricas y variables continuas las cuales deberán clasificarse para poder terminar si un ciudadano entra en la categoría de ingresos mayor a 50k dólares o menor a 50k dólares.
# 

# ## 3. Información del Dataset: "Adult income" y algoritmos para resolver el problema
# 
# ### "Adult income" dataset
# 
# El dataset "Adult income" nos provee información personal de personas viviendo en los Estados Unidos, como la edad, el tipo de trabajo, la educación, cantidad de años de educación formal, estatus civil, raza, etc. y también nos provee de la información de su ingreso anual. El ingreso anual, puede ser mayor o menor a $50,000. 
# 
# El dataset puede utilizarse para que un algoritmo, de acuerdo a la información personal de la persona, pueda aprender a clasificar o predecir, si la persona gana o no, más de $50,000 anuales. Siendo por tanto, el problema que nos presenta este dataset, un problema de clasificación binario. 
# 
# ### Algoritmos que pueden utilizarse
# 
# Para resolver este problema de clasificación, se ha decidido utilizar algunos algoritmos bien conocidos los cuáles son:
# 
# - Árboles de decisión: Un algoritmo sencillo que genera distintos intervalos o condiciones para clasificar cada elemento de acuerdo al valor en sus propiedades.
# - Regresión logística: Utiliza una función de error logística entre la predicción y el valor real de cada ejemplo. El objetivo del algoritmo es minimizar dicho error logístico.
# - Regresión lineal: Se puede utilizar para clasificar si se ajustan los valores de las predicciones. Este algoritmo busca aproximar una función al reducir el error entre la predicción y el valor real de la función.
# - KNN: Este es un método de clasificación no paramétrico, que estima el valor de la función de densidad de probabilidad o directamente la probabilidad a posteriori de que un elemento pertenezca a cierta clase a partir de la información proporcionada por el conjunto de prototipos. 
# - Bayes: Algoritmo para clasificar cierto elemento a partir de probabilidades. 
# 

# ## 4. De los algoritmos seleccionados para resolver el problema deberá de describir justificando: ventaja entre cada uno de ellos, recomendación de elección. 

# ### Ventajas de los algoritmos seleccionados
# ### Regresion lineal
# - Facil de implementar, interpretar y entrenar
# - Es posible extrapolar datos que no estan el en Dataset
# ### Regresion logistica
# - Facil de implementar, interpretar y entrenar
# - Puede interpretar los coeficientes del modelo como indicadores de importancia de características
# ### Naive Bayes
# - Tiene una base matematica solida y una eficiencia de clasificacion estable
# - No es muy sensible a los datos faltantes
# ### Arboles de decision
# - Facil de visualizar e interpretar
# - Puede procesar variables categoricas y continuas simultaneamente
# ### KNN
# - Facil de implementar y practicamente no requiere de entrenamiento
# - Se pueden agregar datos nuevos al modelo facilmente
# 
# ### 4.1. Recomendacion de eleccion
# 
# Se escogieron cuatro modelos utlizados comunmente en problemas de clasificacion: Regresion logistica, Naive Bayes, Arboles de dicision y KNN, y un modelo que no es apropiado utilizar para esta clase de problemas: Regresion lineal. Se escogieron estos modelos para demostrar la importancia de seleccionar un modelo que sea apropiado para la clase de problema que se quiere solucionar, esperando una diferencia considerable en el rendimiento entre la Regresion Lineal y el resto de modelos.

# ## 5. Limites de rendimiento optimo
# Limites de rendimiento optimo.
# Son pruebas que se realizan, desde una perspectiva, para determinar lo rápido que realiza una tarea un sistema en condiciones particulares de trabajo. También puede servir para validar y verificar otros atributos de la calidad del sistema, tales como la escalabilidad, fiabilidad y uso de los recursos. 
# Es muy utilizado en el análisis informatico para optimizar y mejorar el rendimiento optimo del software ante un estudio de datos que se necesitan verificar y analizar.
# Si se cuenta con un volumen alto de datos y el numero de veces que se han ejecutado es necesario estimar el limite para un mejor rendimiento de software y tomar en cuenta el numero de veces que se han vuelto a calcular o generar los datos.
# 

# ## Análisis exploratorio de los datos

# Instalación y carga de dependencias

# In[ ]:


get_ipython().system('pip install plotly;')
get_ipython().system('pip install matplotlib;')
get_ipython().system('pip install chart-studio;')
get_ipython().system('pip install cufflinks;')
get_ipython().system('pip install sklearn;')
get_ipython().system('pip install category_encoders;')
get_ipython().system('pip install graphviz;')


# In[437]:


import numpy as np
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go

from chart_studio.plotly import plot, iplot as py
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots

import cufflinks
cufflinks.go_offline()
init_notebook_mode()

import plotly.io as pio
pio.renderers.default = 'iframe'

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Cargamos nuestro archivo csv

# In[438]:


df = pd.read_csv("adult_income.csv")


# Analizamos el contenido de nuestro data frame

# In[439]:


print(df.shape)
df.columns


# In[440]:


df.head()


# In[441]:


df.info


# Se reemplaza los datos incompletos con NaN (Null) y se observa la cantidad de datos nulos

# In[442]:


df.replace("?",np.nan,inplace = True)
df.isna().sum()


# Como la cantidad de datos nulos es pequeña comparada con el tamaño del dataset, se procede reemplazarlos por la moda de cada columna

# In[443]:


df["workclass"].describe()


# In[444]:


df["workclass"] = df["workclass"].fillna("Private")
df["workclass"].describe()


# In[445]:


df["occupation"].describe()


# In[446]:


df["occupation"] = df["occupation"].fillna("Prof-specialty")
df["occupation"].describe()


# In[447]:


df["native.country"].describe()


# In[448]:


df["native.country"] = df["native.country"].fillna("United-States")
df["native.country"].describe()


# ### Gráficos para facilitar el análisis de datos

# Se puede observar en la gráfica que no hay demasiados datos en los grados de 12th para abajo, por lo que se pueden combinar en un valor

# In[449]:


import matplotlib.pyplot as plt
plt.hist(df["education"])
plt.xticks(rotation='vertical')


# Se procede a combinar los grados de 12th para abajo para simplificar los datos

# In[450]:


df["education"].replace([ "12th","11th","10th", "9th", "7th-8th", "5th-6th", "1st-4th", "Preschool"],
                             "School", inplace = True)
plt.hist(df["education"])
plt.xticks(rotation='vertical')


# Se puede observar en la gráfica que la gran mayoría es de raza blanca y las demás no tienen una cantidad significativa se pueden combinar

# In[451]:


df['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],' Other', inplace = True)
plt.hist(df["race"])
plt.xticks(rotation='vertical')


# Como en los otros casos hay un valor con muchos más datos, por lo que se combinan los demás.

# In[452]:


plt.hist(df["native.country"])
plt.xticks(rotation='vertical')


# Se saca una lista de los países y se borra a USA

# In[453]:


paises = np.array(df["native.country"].unique())
paises = np.delete(paises,np.where(paises=="United-States"))


# Se combina los países diferentes a USA

# In[454]:


df["native.country"].replace(paises, "Other", inplace = True)
plt.hist(df["native.country"])
plt.xticks(rotation='vertical')


# ### Gráficas de atributos después manipulación y limpieza de datos

# Histogramas

# In[455]:


col = 0
NUMBER_OF_COL = 5
row = 0
NUMBER_OF_ROW = 3
i = 0
columns = df.columns

fig, axis = plt.subplots(NUMBER_OF_ROW, NUMBER_OF_COL, figsize=(15,15))

for row in range(NUMBER_OF_ROW):
    for col in range(NUMBER_OF_COL):
        axis[row,col].hist(df[columns[i]])
        axis[row,col].set_title(columns[i])
        axis[row,col].tick_params(axis='x', labelrotation=90)
        i += 1 
fig.tight_layout()
plt.show()


# Aquí, en este histograma, se puede observar ciertas cuestiones curiosas en la distribución de los datos. Quizas el más importante es que el dataset no está balanceado, muchos más ejemplos de personas con un ingreso menor de \\$50,000 y pocos con más de \\$50,000. 
# 
# También es fácil darse cuenta que la mayoría de los datos son de personas blancas, de sexo masculino, casados, profesionales, con aproximádamente 10 años de educación formal (con diploma de secundaria y algunos estudios universitarios, pero sin completar), trabajando en el sector privado y con una edad de entre 20 y 50 años.

# Visualizacion de variables continuas por medio de grafico de caja, para comprender las medidas de tendencia central.

# In[456]:


fig = go.Figure()
fig = make_subplots(rows=2, cols=3, start_cell="bottom-left")
fig.add_trace(go.Box(x=df['hours.per.week'],name="hours.per.week"),
              row=1, col=1)

fig.add_trace(go.Box(x=df['age'],name="Edad"),
              row=1, col=2)

fig.add_trace(go.Box(x=df['fnlwgt'],name="fnlwgt"),
              row=1, col=3)
fig.add_trace(go.Box(x=df['education.num'],name="education.num"),
              row=2, col=1)

fig.add_trace(go.Box(x=df['capital.loss'],name="capital.loss"),
              row=2, col=2)

fig.add_trace(go.Box(x=df['capital.gain'],name="capital.gain"),
              row=2, col=3)
fig.show()


# Mapa de calor mostrando la relación entre los coeficientes de correlación entre features y etiquetas

# In[457]:


fig = px.imshow(df.corr(), text_auto=True)
fig.update_layout(height=700, width=700)
fig.show()


# Método de Bayes
# 
# El teorema de bayes se puede utilizar para calcular la probabilidad condicional. Para nuestro estudio haremos uso de Naive Bayes la cual es una colección de algoritmos de clasificación para machine learning basado en el teorema de Bayes. Es una simple técnica de clasificación, pero tiene alta funcionalidad. En nuestro estudio utilizaremos Gaussian Naive Bayes de Naive Bayes
# 
# Analizamos si tenemos variables categóricas

# In[458]:


categoricas = [var for var in df.columns if df[var].dtype=='O']
print('Existen {} variables categóricas\n'.format(len(categoricas)))

print('Las cuales son:\n\n', categoricas)


# Se determino que existen 8 variables categóricas y 1 binaria las restantes son variables continuas.
# Eliminamos income de nuestro set de entrenamiento x y creamos el set y con dichos valores.

# Cargamos nuestro modelo de selección 

# In[459]:


from sklearn.model_selection import train_test_split


# Codificamos nuestras variables categóricas para que puedan ser tratadas como numéricas

# In[460]:


X = df.loc[:, df.columns != 'income']
Y = df['income']


# In[461]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[462]:


from sklearn import preprocessing

categorical = ['workclass','education', 'marital.status', 'occupation', 'relationship','race', 'sex','native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# Cargamos nuestro modelo Gaussian Naive Bayes

# In[463]:


from sklearn.naive_bayes import GaussianNB


# Alimentamos nuestro algoritmo

# In[464]:


gnb = GaussianNB()

gnb.fit(X_train, y_train)


# Realizamos nuestra predicción 

# In[465]:


y_pred = gnb.predict(X_test)

y_pred


# Calculamos la precisión de nuestro modelo 

# In[466]:


from sklearn.metrics import accuracy_score

print('Precisión del modelo: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[467]:


print('Puntuacion set de entrenamiento: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Puntuacion set de prueba: {:.4f}'.format(gnb.score(X_test, y_test)))

acc_GN = round(gnb.score(X_train, y_train) * 100, 2)


# Visualizamos la matriz de confusión para ver el desempeño del modelo 

# | Predicted Class |  |  |  |
# | --- | --- | --- | --- |
# ||  | Class = Yes | Class = No |
# | Actual Class  | Class = Yes | True Positive | False Negative |
# | | Class = No | False Positive | True Negative |

# In[468]:


from sklearn.metrics import confusion_matrix  
cm_GN = confusion_matrix(y_test, y_pred, labels=gnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_GN,display_labels=gnb.classes_)
disp.plot()
plt.show()


# In[469]:


y_pred_gn = y_pred
y_test_gn = y_test


# ## Regresión Logística 

# In[470]:


from sklearn.linear_model import LogisticRegression


# In[471]:



logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log


# In[472]:


cm_log = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log,display_labels=logreg.classes_)
disp.plot()
plt.show()


# ## Aplicación del algoritmo Decision Tree

# In[473]:


columnas_a_codificar = ['workclass','education', 'marital.status', 'occupation', 
                        'relationship', 'race', 'native.country', 'income']

df[columnas_a_codificar] = df[columnas_a_codificar].apply(lambda col:pd.Categorical(col).codes)
df


# In[474]:


X = df.loc[:, df.columns != 'income']
Y = df['income']


# In[475]:


from sklearn import tree
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[476]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[477]:


###acc_clf = clf.score(X_test, y_test)
acc_clf = round(clf.score(X_test, y_test) * 100, 2)
acc_clf


# In[478]:


y_pred = clf.predict(X_test)
cm_clf = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_clf,display_labels=clf.classes_)
disp.plot()
plt.show()


# También se puedo visualizar el árbol de decisión:

# In[479]:


import graphviz 
df_with_strings = pd.read_csv('adult_income.csv')
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=X.columns,  
                      class_names=df_with_strings.income.unique(),  
                      filled=True, rounded=True,  
                      special_characters=False) 
graph = graphviz.Source(dot_data) 
graph.render("adult_income") 


# ## Método lineal 

# In[480]:


columnas_a_codificar = ['workclass','education', 'marital.status', 'occupation', 
                        'relationship', 'race', 'native.country', 'income']

df[columnas_a_codificar] = df[columnas_a_codificar].apply(lambda col:pd.Categorical(col).codes)
df


# In[481]:


X = df.loc[:, df.columns != "income"]
Y = df["income"]


# In[482]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)


# In[483]:


from sklearn.preprocessing import PolynomialFeatures
x_ = PolynomialFeatures(degree=2).fit_transform(X_train)
model2 = LinearRegression().fit(x_, y_train)
model2.score(x_, y_train)


# In[484]:


x_ = PolynomialFeatures(degree=3).fit_transform(X_train)
model3 = LinearRegression().fit(x_, y_train)
model3.score(x_, y_train)
acc_LR = round(model3.score(x_, y_train) * 100, 2)
acc_LR


# In[485]:


import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[486]:


error = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    #print("accuracy: {}".format(accuracy_score(y_test, pred)))

grafica = px.line(x=range(1, 31),y=error,  markers=True)
grafica.show()


# In[487]:


k_optimo = list(range(1,31))[error.index(min(error))]
k_optimo


# In[488]:


acc_KNN = (1-error[27])* 100
acc_KNN


# ## Resultados

# In[489]:


results = pd.DataFrame({
    'Model': [ 'KNN', 'Regresión Lineal', 'Regresión logística', 'Naive Bayes', 'Arboles de decision'],
    'Score': [acc_KNN, acc_LR,  acc_log, acc_GN, acc_clf]})
result_df = results.sort_values(by='Score', ascending=False)
##result_df = result_df.set_index('Score')
result_df.head(7)


# ## 6. Justificación de los intervalos de confianza utilizados

# Se tomo la decisión de utilizar un intervalo de confianza con un nivel de significancia del 5%, debido a que se deseaba tener mayor certeza en la estimación de la muestra respecto al valor poblacional. 

# In[490]:


import math
def intervalo_Confianza(z,accuracy,nMuestra):
    return z * math.sqrt(accuracy * (1-(accuracy/100))/nMuestra)


# In[491]:


result_df["Intervalos de confianza"]=result_df["Score"].apply(lambda x: intervalo_Confianza(1.96,float(x),len(X_test))*100)


# In[492]:


result_df.head(7)


# ## 7. Análisis de la estimación de la tasa de éxito y de la estimación de error

# Para estimar el éxito de nuestros modelos se analizará el accuracy obtenido por nuestro set de pruebas para cada uno de estos, este dato se será convertido en porcentaje y se estimará el error como la diferencia entre la precisión de un modelo ideal y la obtenida por cada uno de estos.
# 
# Estos datos son soportados por medio de matrices de confusión, las cuales nos muestran de una manera grafica como están distribuidos los resultados.
# 
# | Predicted Class |  |  |  |
# | --- | --- | --- | --- |
# ||  | Class = Yes | Class = No |
# | Actual Class  | Class = Yes | True Positive | False Negative |
# | | Class = No | False Positive | True Negative |
# 
# 

# In[493]:


result_df["Error"]=100-result_df["Score"]
result_df[['Model','Score','Error','Intervalos de confianza']]


# ## 8. Análisis, fundamento y descripción de la muestra de entrenamiento

# La muestra de entrenamiento utilizada en los modelos realizad, se abordó de manera tal que, luego de realizar una limpieza y manipulación al conjunto de datos originales se destinó un porcentaje de 67% para la misma. La elección de este porcentaje se fundamentó en que la predicción del modelo no se viese afecta y colocara al modelo en dos posibles situaciones.
# La primera situación por evitar es que el modelo posea un ajuste insuficiente y no pueda capturar la tendencia de los datos. Por ello se escogió un porcentaje de 67%, no menos, de los datos para el entrenamiento. La segunda situación que evitar es que el modelo tenga un sobreajuste, en el cual se caracteriza porque modelo intenta cubrir demasiado puntos, lo cual introduce ruido al modelo y comienza a predecir erróneamente afectando de gran manera la eficiencia y precisión de este. Por esta razón se decidió utilizar un 33%, no menos, de los datos para realizar pruebas y encontrar modelos ajustados con un puntaje considerablemente bueno.

# ## 9. Planteamiento de la hipótesis del problema a resolver

# H0:
# Se puede predecir si el ingreso de una persona supera los $50,000 anuales en función de características como: 'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country' con un minimo del 80% de exactitud.
# 
# H1: No se puede predecir.

# ## 10. Interpretación de los resultados obtenidos

# De acuerdo a los resultados obtenidos en la tabla [201] se pudo observar que el modelo regresión logística obtuvo la mayor exactitud con un 82.48% $\pm$ 7.18 con un error de 17.52%, siendo este el modelo con mejor rendimiento para nuestro conjunto de datos. Al contrario, el modelo de regresión lineal presento una exactitud de 44.51 $\pm$ 9.4 y con un error de 55.49% consideramos que esto se debe a que este algoritmo se utiliza principalmente para predicción de variables continuas, por lo que no se acopla debidamente al problema ya que este es de clasificación.
# 
# El resto de los modelos KNN, arboles de decisión y naive bayes muestra un rendimiento muy similar entre ellos tanto en su precisión como en su error. La exactitud de estos modelos oscila entre el 79.63% y 81.29%.

# ## 11. Sus conclusiones deberán ser soportadas por gráficos y anotaciones en los gráficos que se utilicen

# Haciendo uso de algoritmos de machine learning y después de entrenar y probar dichos modelos, se puede concluir que es posible determinar la categoría de ingresos que poseerá un adulto con exactitud del 80%. Por lo que se aprueba la hipótesis nula, ya que fue posible predecir dicha clasificación con el modelo de regresión logística con una exactitud del 82.48% $\pm$ 7.18 con un error de 17.52% y se rechaza la hipótesis alternativa.
# 

# In[494]:


fig = px.scatter(result_df,x='Model',y='Score',error_y='Intervalos de confianza',title="Comparación de algoritmos")
fig.show()


# Como se observa en el grafico anterior, la mayoria de modelos presenta una exactitud cercana al 80% a excepcion del regresion lineal ya que este algoritmo no es recomendado para realizar tareas de clasificacion.  
