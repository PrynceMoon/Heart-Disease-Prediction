# %% 
# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Loading the dataset
df = pd.read_csv('C:/Users/anton/Desktop/Q-learning-master/heart.csv')

# %%
# Exploring the dataset
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# %%
# Check for missing values and basic info
print(df.isnull().any())
df.info()
print(df.describe().T)

# %%
# Data Visualization
plt.figure(figsize=(15,15))
df.hist()
plt.show()

# %%
g = sns.countplot(x='target', data=df)
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# %%
# Feature Engineering: Heatmap for correlation
corr_matrix = df.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(20,20))
sns.heatmap(data=df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()

# %%
# Data Preprocessing
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# %%
from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])

# %%
# Splitting the dataset into dependent and independent features
X = dataset.drop('target', axis=1)
y = dataset['target']

# %%
# Model Building: KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_scores = []
for i in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    cvs_scores = cross_val_score(knn_classifier, X, y, cv=10)
    knn_scores.append(round(cvs_scores.mean(), 3))

plt.figure(figsize=(20,15))
plt.plot([k for k in range(1, 21)], knn_scores, color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=12)
cvs_scores = cross_val_score(knn_classifier, X, y, cv=10)
print("KNeighbors Classifier Accuracy with K=12 is: {}%".format(round(cvs_scores.mean(), 4)*100))

# %%
# Model Building: Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

decision_scores = []
for i in range(1, 11):
    decision_classifier = DecisionTreeClassifier(max_depth=i)
    cvs_scores = cross_val_score(decision_classifier, X, y, cv=10)
    decision_scores.append(round(cvs_scores.mean(), 3))

plt.figure(figsize=(20,15))
plt.plot([i for i in range(1, 11)], decision_scores, color='red')
for i in range(1, 11):
    plt.text(i, decision_scores[i-1], (i, decision_scores[i-1]))
plt.xticks([i for i in range(1, 11)])
plt.xlabel('Depth of Decision Tree (N)')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different depth values')
plt.show()

decision_classifier = DecisionTreeClassifier(max_depth=3)
cvs_scores = cross_val_score(decision_classifier, X, y, cv=10)
print("Decision Tree Classifier Accuracy with max_depth=3 is: {}%".format(round(cvs_scores.mean(), 4)*100))

# %%
# Model Building: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest_scores = []
for i in range(10, 101, 10):
    forest_classifier = RandomForestClassifier(n_estimators=i)
    cvs_scores = cross_val_score(forest_classifier, X, y, cv=5)
    forest_scores.append(round(cvs_scores.mean(), 3))

plt.figure(figsize=(20,15))
plt.plot([n for n in range(10, 101, 10)], forest_scores, color='red')
for i in range(1, 11):
    plt.text(i*10, forest_scores[i-1], (i*10, forest_scores[i-1]))
plt.xticks([i for i in range(10, 101, 10)])
plt.xlabel('Number of Estimators (N)')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different N values')
plt.show()

forest_classifier = RandomForestClassifier(n_estimators=90)
cvs_scores = cross_val_score(forest_classifier, X, y, cv=5)
print("Random Forest Classifier Accuracy with n_estimators=90 is: {}%".format(round(cvs_scores.mean(), 4)*100))

# %% [markdown]
# ### Visualizzazione dei risultati delle query Prolog
# Alla fine dell’esecuzione del programma, integriamo i due file Prolog per eseguire le query
# definite in *heart_rules.pl* che acquisiscono i dati dal file *heart_kb.pl*.
#
# Il risultato verrà mostrato con il seguente template:
#
# 📊 Analisi basata su regole Prolog:
#
# 🔴 Pazienti ad ALTO RISCHIO: [{'Età': 63, 'Sesso': 1}, {'Età': 57, 'Sesso': 0}, ...]
# 🟠 Pazienti con IPERTENSIONE o COLESTEROLO ALTO: [{'Età': 52, 'Sesso': 1}, {'Età': 59, 'Sesso': 1}, ...]
# 🔵 Pazienti con ECG ANOMALO: [{'Età': 37, 'Sesso': 1}, {'Età': 56, 'Sesso': 0}, ...]
# ⚠️ Pazienti con ANGINA DA SFORZO: [{'Età': 57, 'Sesso': 1}, {'Età': 62, 'Sesso': 1}, ...]
# 🔥 Pazienti con PROFILO AD ALTO RISCHIO: [{'Età': 63, 'Sesso': 1}, {'Età': 57, 'Sesso': 0}, ...]
#
# Assicurati che nei file Prolog i predicati restituiscano una struttura che contenga almeno i dati relativi ad
# "Età" e "Sesso". Potrebbe trattarsi, ad esempio, di un termine del tipo: patient(Età, Sesso, …).

# %%
from pyswip import Prolog

# Inizializza l'interprete Prolog
prolog = Prolog()

# Consulta il file heart_rules.pl (che al suo interno include anche heart_kb.pl)
prolog.consult("heart_rules.pl")

# Esegui le query definite in heart_rules.pl.
# Ciascuna query restituisce un dizionario con le variabili corrispondenti.
results_alto_rischio = [ {"Età": res["Età"], "Sesso": res["Sesso"]} 
                         for res in prolog.query("alto_rischio(Età, Sesso)") ]

results_ipertensione = [ {"Età": res["Età"], "Sesso": res["Sesso"]} 
                         for res in prolog.query("ipertensione_colesterolo(Età, Sesso)") ]

results_ecg_anomalo = [ {"Età": res["Età"], "Sesso": res["Sesso"]} 
                        for res in prolog.query("ecg_anomalo(Età, Sesso)") ]

results_angina = [ {"Età": res["Età"], "Sesso": res["Sesso"]} 
                   for res in prolog.query("angina_sforzo(Età, Sesso)") ]

results_profilo = [ {"Età": res["Età"], "Sesso": res["Sesso"]} 
                    for res in prolog.query("profilo_alto_rischio(Età, Sesso)") ]

# Visualizza i risultati formattati secondo il template richiesto
print("\n📊 Analisi basata su regole Prolog:\n")
print("🔴 Pazienti ad ALTO RISCHIO:", results_alto_rischio)
print("🟠 Pazienti con IPERTENSIONE o COLESTEROLO ALTO:", results_ipertensione)
print("🔵 Pazienti con ECG ANOMALO:", results_ecg_anomalo)
print("⚠️ Pazienti con ANGINA DA SFORZO:", results_angina)
print("🔥 Pazienti con PROFILO AD ALTO RISCHIO:", results_profilo)
