# %% 
# Importazione delle librerie essenziali
import numpy as np                     # Operazioni numeriche avanzate
import pandas as pd                    # Manipolazione dei dati in DataFrame
import matplotlib.pyplot as plt        # Creazione di grafici e visualizzazioni
import seaborn as sns                  # Visualizzazione dei dati con stili predefiniti
from pyswip import Prolog              # Interfaccia per utilizzare Prolog in Python
from collections import deque         # Struttura dati "deque" per operazioni di coda (BFS)
# Import delle funzioni per calcolare le metriche di valutazione dei modelli
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Funzione per dividere il dataset in set di training e test
from sklearn.model_selection import train_test_split

# %%
# Caricamento del dataset
# Legge il file CSV contenente i dati relativi a malattie cardiache
dati = pd.read_csv('C:/Users/anton/Desktop/Q-learning-master/heart.csv')

# %%
# Esplorazione del dataset
# Stampa la forma (numero di righe e colonne), i nomi delle colonne e le prime 5 righe
print("Forma del dataset:", dati.shape)
print("Colonne:", dati.columns.tolist())
print(dati.head())

# %%
# Controllo dei valori mancanti e informazioni di base
# Verifica la presenza di valori nulli per ogni colonna
print(dati.isnull().any())
# Mostra informazioni dettagliate sul DataFrame (tipi di dato, numero di valori non nulli, memoria utilizzata)
dati.info()
# Calcola e stampa statistiche descrittive (media, deviazione standard, min, max, etc.) trasposte per leggibilità
print(dati.describe().T)

# %%
# Visualizzazione dei dati
# Crea una figura di dimensioni 15x15 pollici e genera istogrammi per ogni variabile nel dataset
plt.figure(figsize=(15,15))
dati.hist()
plt.show()  # Mostra i grafici

# %%
# Visualizzazione della distribuzione della variabile target
# Crea un grafico a barre che mostra il numero di osservazioni per ciascuna classe del target
g = sns.countplot(x='target', data=dati)
plt.xlabel('Obiettivo')   # Etichetta asse x
plt.ylabel('Conteggio')   # Etichetta asse y
plt.show()                # Mostra il grafico

# %%
# Ingegneria delle caratteristiche: Heatmap per la correlazione
# Calcola la matrice di correlazione tra le variabili
matrice_corr = dati.corr()
# Ottiene l'indice della matrice, cioè i nomi delle variabili
caratteristiche_corr = matrice_corr.index
# Crea una figura ampia per visualizzare la heatmap
plt.figure(figsize=(20,20))
# Visualizza la matrice di correlazione con annotazioni e una mappa di colori 'RdYlGn'
sns.heatmap(data=dati[caratteristiche_corr].corr(), annot=True, cmap='RdYlGn')
plt.show()

# %%
# Pre-elaborazione dei dati
# Trasforma le variabili categoriche in variabili dummy (one-hot encoding)
dataset = pd.get_dummies(dati, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# %%
# Standardizzazione delle variabili numeriche
from sklearn.preprocessing import StandardScaler
scalatore = StandardScaler()
# Lista delle colonne numeriche da standardizzare
colonne_da_scalare = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# Applica lo scaling: trasforma i dati in modo che abbiano media 0 e deviazione standard 1
dataset[colonne_da_scalare] = scalatore.fit_transform(dataset[colonne_da_scalare])

# %%
# Suddivisione del dataset in variabili indipendenti (X) e variabile dipendente (y)
# X contiene tutte le colonne eccetto 'target'
X = dataset.drop('target', axis=1)
# y contiene solo la colonna 'target'
y = dataset['target']

# %% 
# Suddivisione in set di training e test
# Divide il dataset in 80% training e 20% test con random_state fissato per riproducibilità
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Funzione per calcolare e stampare le metriche di valutazione
def stampa_metriche(y_test, y_pred, nome_modello):
    """
    Calcola e stampa diverse metriche di valutazione per un modello di classificazione.
    
    Parametri:
        y_test: valori reali del target del set di test.
        y_pred: predizioni generate dal modello.
        nome_modello: stringa che identifica il modello.
    """
    # Calcola l'accuratezza (percentuale di predizioni corrette)
    accuracy = accuracy_score(y_test, y_pred)
    # Calcola la precisione (frazione di predizioni positive corrette)
    precision = precision_score(y_test, y_pred)
    # Calcola il recall (frazione di veri positivi identificati)
    recall = recall_score(y_test, y_pred)
    # Calcola il f1-score (media armonica di precision e recall)
    f1 = f1_score(y_test, y_pred)
    # Calcola l'area sotto la curva ROC (capacità del modello di distinguere le classi)
    roc_auc = roc_auc_score(y_test, y_pred) 
    
    # Stampa le metriche formattate
    print(f"\nMetriche per {nome_modello}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

# %% 
# Costruzione del modello: Classificatore del Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Calcola i punteggi medi per diversi valori di profondità dell'albero (da 1 a 10)
punteggi_albero = []
for i in range(1, 11):
    # Crea un modello temporaneo con max_depth pari a i
    modello_temp = DecisionTreeClassifier(max_depth=i)
    # Esegue la validazione incrociata a 10 fold e calcola il punteggio medio
    punteggi_cv = cross_val_score(modello_temp, X, y, cv=10)
    punteggi_albero.append(round(punteggi_cv.mean(), 3))

# Visualizza graficamente come varia la performance in funzione della profondità
plt.figure(figsize=(20,15))
plt.plot([i for i in range(1, 11)], punteggi_albero, color='red')
# Aggiunge il testo con il punteggio per ogni profondità
for i in range(1, 11):
    plt.text(i, punteggi_albero[i-1], (i, punteggi_albero[i-1]))
plt.xticks([i for i in range(1, 11)])
plt.xlabel('Profondità del Decision Tree (N)')
plt.ylabel('Punteggi')
plt.title('Punteggi del classificatore Decision Tree per diversi valori di profondità')
plt.show()

# Addestramento del Decision Tree con max_depth=3
# Si sceglie max_depth=3 in base ai risultati ottenuti dalla cross-validazione
classificatore_albero = DecisionTreeClassifier(max_depth=3)
# Addestra il modello sui dati di training
classificatore_albero.fit(X_train, y_train)
# Predice il target sui dati di test
y_pred_albero = classificatore_albero.predict(X_test)
# Calcola e stampa le metriche di valutazione per il modello Decision Tree
stampa_metriche(y_test, y_pred_albero, "Decision Tree")

# %%
# Costruzione del modello: Classificatore Random Forest
from sklearn.ensemble import RandomForestClassifier

# Calcola i punteggi medi per diversi valori del numero di stimatori (n_estimators)
punteggi_foresta = []
for i in range(10, 101, 10):
    # Crea un modello temporaneo con n_estimators pari a i
    modello_temp = RandomForestClassifier(n_estimators=i)
    # Esegue la validazione incrociata a 5 fold e calcola il punteggio medio
    punteggi_cv = cross_val_score(modello_temp, X, y, cv=5)
    punteggi_foresta.append(round(punteggi_cv.mean(), 3))

# Visualizza graficamente la performance in funzione del numero di stimatori
plt.figure(figsize=(20,15))
plt.plot([n for n in range(10, 101, 10)], punteggi_foresta, color='red')
for i in range(1, 11):
    plt.text(i*10, punteggi_foresta[i-1], (i*10, punteggi_foresta[i-1]))
plt.xticks([i for i in range(10, 101, 10)])
plt.xlabel('Numero di Stimatori (N)')
plt.ylabel('Punteggi')
plt.title('Punteggi del classificatore Random Forest per diversi valori di N')
plt.show()

# Addestramento della Random Forest con n_estimators=90
# Si sceglie n_estimators=90 in base ai risultati ottenuti dalla cross-validazione
classificatore_foresta = RandomForestClassifier(n_estimators=90)
# Addestra il modello sui dati di training
classificatore_foresta.fit(X_train, y_train)
# Predice il target sui dati di test
y_pred_foresta = classificatore_foresta.predict(X_test)
# Calcola e stampa le metriche di valutazione per il modello Random Forest
stampa_metriche(y_test, y_pred_foresta, "Random Forest")

# %% [markdown]
# ### Visualizzazione dei risultati delle query Prolog
# In questa sezione integriamo i file Prolog per eseguire query basate su regole
# definite nei file 'heart_rules.pl' (che include anche 'heart_kb.pl').
# I risultati verranno mostrati con un template che evidenzia le diverse categorie di pazienti.

# %%
# Inizializza l'interprete Prolog e carica il file di regole
prolog = Prolog()
prolog.consult("heart_rules.pl")

# Esegue la query "alto_rischio" e raccoglie i risultati in una lista di dizionari
risultati_alto_rischio = [{"Età": res["Età"], "Sesso": res["Sesso"]} 
                          for res in prolog.query("alto_rischio(Età, Sesso)")]
# Esegue la query "ipertensione_colesterolo" e raccoglie i risultati
risultati_ipertensione = [{"Età": res["Età"], "Sesso": res["Sesso"]} 
                          for res in prolog.query("ipertensione_colesterolo(Età, Sesso)")]
# Esegue la query "ecg_anomalo" e raccoglie i risultati
risultati_ecg_anomalo = [{"Età": res["Età"], "Sesso": res["Sesso"]} 
                         for res in prolog.query("ecg_anomalo(Età, Sesso)")]
# Esegue la query "angina_sforzo" e raccoglie i risultati
risultati_angina = [{"Età": res["Età"], "Sesso": res["Sesso"]} 
                   for res in prolog.query("angina_sforzo(Età, Sesso)")]
# Esegue la query "profilo_alto_rischio" e raccoglie i risultati
risultati_profilo = [{"Età": res["Età"], "Sesso": res["Sesso"]} 
                    for res in prolog.query("profilo_alto_rischio(Età, Sesso)")]

# Funzione per limitare il numero di risultati mostrati
def tronca_risultati(risultati, max_elementi=3):
    """
    Limita la visualizzazione dei risultati a 'max_elementi'.
    Se il numero totale di risultati è maggiore di 'max_elementi',
    vengono mostrati i primi (max_elementi - 1) e un segnaposto "{...,...}".
    """
    if len(risultati) > max_elementi:
        return risultati[:max_elementi-1] + ["{...,...}"]
    return risultati

# Stampa i risultati delle query Prolog con il template richiesto
print("\n📊 Analisi basata su regole Prolog:\n")
print("🔴 Pazienti ad ALTO RISCHIO:", tronca_risultati(risultati_alto_rischio))
print("🟠 Pazienti con IPERTENSIONE o COLESTEROLO ALTO:", tronca_risultati(risultati_ipertensione))
print("🔵 Pazienti con ECG ANOMALO:", tronca_risultati(risultati_ecg_anomalo))
print("⚠️ Pazienti con ANGINA DA SFORZO:", tronca_risultati(risultati_angina))
print("🔥 Pazienti con PROFILO AD ALTO RISCHIO:", tronca_risultati(risultati_profilo))

# %%
# Algoritmo BFS per trovare pazienti simili
def bfs_trova_pazienti_simili(eta_iniziale, sesso_iniziale, profondita_max=3):
    """
    Implementa la ricerca in ampiezza (BFS) per trovare pazienti simili.
    
    Parametri:
      - eta_iniziale: età del paziente iniziale
      - sesso_iniziale: sesso del paziente iniziale
      - profondita_max: profondità massima della ricerca (default 3)
    
    La funzione utilizza Prolog per trovare pazienti che soddisfano una certa condizione,
    in questo caso definita dalla query 'heart_patient'. La ricerca si interrompe se la profondità
    supera 'profondita_max' o se il paziente è già stato visitato. Infine, restituisce i primi due
    risultati e, se ce ne sono più di tre, aggiunge un segnaposto di troncamento.
    """
    # Inizializza Prolog e carica il file di regole
    prolog = Prolog()
    prolog.consult("heart_rules.pl")
    
    # Inizializza la coda per la BFS con il paziente iniziale e profondità 0
    coda = [(eta_iniziale, sesso_iniziale, 0)]
    # Insieme per tenere traccia dei pazienti già visitati (per evitare duplicazioni)
    visitati = set()
    # Lista per raccogliere i pazienti simili trovati
    pazienti_simili = []
    
    # Esegue la BFS finché la coda non è vuota
    while coda:
        # Estrae il primo elemento dalla coda (FIFO)
        eta, sesso, profondita = coda.pop(0)
        
        # Se la profondità corrente supera il limite massimo, interrompe la ricerca
        if profondita > profondita_max:
            break
        
        # Se il paziente è già stato visitato, salta al prossimo
        if (eta, sesso) in visitati:
            continue
        
        # Aggiunge il paziente all'insieme dei visitati
        visitati.add((eta, sesso))
        # Aggiunge il paziente alla lista dei pazienti simili
        pazienti_simili.append({"Età": eta, "Sesso": sesso})
        
        # Esegue una query Prolog per trovare pazienti simili (basato su una condizione ipotetica)
        for res in prolog.query(f"heart_patient(A, {sesso}, _, B, _, _, _, _, _, _, _, _, _, _)"):
            nuova_eta, nuova_pressione = res["A"], res["B"]
            # Se il paziente trovato ha un'età diversa e una pressione simile (entro ±30 da 120), lo aggiunge alla coda
            if nuova_eta != eta and abs(nuova_pressione - 120) <= 30:
                coda.append((nuova_eta, sesso, profondita + 1))
    
    # Restituisce i primi due pazienti trovati e, se ce ne sono più di tre, aggiunge un segnaposto per indicare troncamento
    return pazienti_simili[:2] + (["{...,...}"] if len(pazienti_simili) > 3 else pazienti_simili)

# Esempio di utilizzo della BFS:
paziente_iniziale = (57, 1)  # Paziente iniziale: 57 anni, sesso 1 (maschio)
pazienti_simili = bfs_trova_pazienti_simili(*paziente_iniziale)
print("Pazienti con caratteristiche simili trovati via BFS:", pazienti_simili)
