# Analisi e Classificazione dei Dati Cardiaci

Questo progetto ha l'obiettivo di analizzare un dataset relativo alla salute cardiaca e di implementare diversi modelli di machine learning per classificare i pazienti in base al rischio cardiovascolare. Inoltre, integra un modulo basato su regole Prolog per eseguire query specifiche sui dati dei pazienti, offrendo così una doppia prospettiva: statistica e logica.

---

## Indice

- [Introduzione](#introduzione)
- [Librerie Utilizzate](#librerie-utilizzate)
- [Caricamento ed Esplorazione del Dataset](#caricamento-ed-esplorazione-del-dataset)
- [Visualizzazione dei Dati](#visualizzazione-dei-dati)
- [Preprocessing dei Dati](#preprocessing-dei-dati)
- [Modelli di Machine Learning](#modelli-di-machine-learning)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
- [Integrazione con Prolog](#integrazione-con-prolog)
- [Esecuzione del Codice](#esecuzione-del-codice)
- [Conclusioni](#conclusioni)

---

## Introduzione

Il codice presente in questo progetto guida l'utente attraverso le seguenti fasi:

1. **Caricamento ed Esplorazione del Dataset:** Vengono esaminate le caratteristiche del dataset, inclusa la verifica di eventuali valori mancanti e l'analisi statistica.
2. **Visualizzazione dei Dati:** Vengono creati vari grafici (istogrammi, countplot e heatmap) per aiutare a comprendere la distribuzione e le correlazioni tra le variabili.
3. **Preprocessing dei Dati:** Le variabili categoriche vengono trasformate tramite one-hot encoding, mentre alcune variabili numeriche vengono normalizzate.
4. **Costruzione ed Evaluazione dei Modelli:** Vengono costruiti tre modelli di machine learning (K-Nearest Neighbors, Decision Tree e Random Forest) utilizzando la validazione incrociata per determinare le impostazioni ottimali.
5. **Integrazione con Prolog:** Utilizzando PySwip, il codice esegue query su un file Prolog contenente regole e una knowledge base per identificare categorie specifiche di pazienti (ad esempio, pazienti ad alto rischio).

---

## Librerie Utilizzate

Il progetto si avvale di numerose librerie Python, tra cui:

- **NumPy & Pandas:** Gestione e manipolazione dei dati.
- **Matplotlib & Seaborn:** Visualizzazione grafica dei dati.
- **Scikit-learn:** Preprocessing, normalizzazione e costruzione dei modelli di machine learning.
- **PySwip:** Integrazione di un interprete Prolog all'interno di Python per l'esecuzione di regole logiche.

---

## Caricamento ed Esplorazione del Dataset

- **Caricamento:** Il dataset `heart.csv` viene importato utilizzando la funzione `pd.read_csv()`.  
  _Nota:_ Assicurarsi che il percorso del file sia corretto in base alla propria configurazione.
  
- **Esplorazione:**  
  - Viene stampata la **forma** (shape) del dataset e la **lista delle colonne**.
  - Viene mostrata una **anteprima** dei dati (`head()`).
  - Viene verificata la presenza di **valori nulli** e vengono fornite informazioni dettagliate (mediante `info()` e `describe()`).

---

## Visualizzazione dei Dati

Per una prima analisi visiva, il codice genera diversi grafici:

- **Istogrammi:** Mostrano la distribuzione delle variabili del dataset.
- **Countplot per la Variabile Target:** Evidenzia il numero di istanze per ciascuna classe del target (indicatore di rischio).
- **Heatmap della Correlazione:** Visualizza la matrice di correlazione tra le variabili per individuare eventuali relazioni significative.

Queste visualizzazioni aiutano a comprendere la struttura dei dati e a identificare eventuali pattern o anomalie.

---

## Preprocessing dei Dati

Prima di procedere alla costruzione dei modelli, il dataset viene opportunamente preprocessato:

1. **One-Hot Encoding:**  
   Le variabili categoriche (ad esempio, `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) vengono convertite in variabili dummy per renderle compatibili con i modelli di machine learning.

2. **Standardizzazione:**  
   Alcune variabili numeriche (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) vengono normalizzate utilizzando il `StandardScaler`, garantendo così che abbiano una scala comune.

3. **Divisione in Feature e Target:**  
   Il dataset viene diviso in:
   - **X:** Tutte le variabili indipendenti (dopo preprocessing).
   - **y:** La variabile dipendente `target` che indica il rischio cardiaco.

---

## Modelli di Machine Learning

Il codice implementa e valuta tre diversi modelli di classificazione, utilizzando la **validazione incrociata** per garantire una stima robusta delle prestazioni.

### K-Nearest Neighbors

- **Obiettivo:**  
  Classificare i pazienti in base al rischio cardiaco utilizzando il principio dei vicini più prossimi.

- **Procedura:**
  1. Viene eseguita una validazione incrociata (10-fold) variando il numero di vicini (`n_neighbors`) da 1 a 20.
  2. I punteggi medi vengono memorizzati in una lista e successivamente plottati per visualizzare l'andamento dell'accuratezza al variare di `K`.
  3. Il modello finale viene addestrato con `K=12`, e viene stampata l'accuratezza ottenuta.

### Decision Tree

- **Obiettivo:**  
  Utilizzare un albero decisionale per segmentare i dati e classificare i pazienti.

- **Procedura:**
  1. Viene variata la profondità dell'albero (`max_depth`) da 1 a 10, utilizzando la validazione incrociata (10-fold).
  2. I punteggi medi per ciascuna profondità vengono plottati, permettendo di identificare il valore ottimale.
  3. Il modello finale viene scelto con `max_depth=3` e l'accuratezza viene stampata.

### Random Forest

- **Obiettivo:**  
  Impiegare una foresta casuale per migliorare la robustezza e la precisione della classificazione.

- **Procedura:**
  1. Il numero di stimatori (`n_estimators`) viene variato da 10 a 100 (step di 10) utilizzando una validazione incrociata (5-fold).
  2. I punteggi medi vengono plottati per osservare come varia l'accuratezza con l'aumentare degli alberi.
  3. Il modello finale viene configurato con `n_estimators=90` e viene stampata l'accuratezza finale.

---

## Integrazione con Prolog

Oltre ai modelli di machine learning, il progetto integra un modulo basato su **Prolog** per analisi logiche e rule-based:

- **PySwip:**  
  Viene inizializzato un interprete Prolog all'interno del codice Python.

- **Consultazione dei File Prolog:**  
  Il file `heart_rules.pl` (che include anche `heart_kb.pl`) viene consultato. Questo file contiene le regole e la knowledge base necessaria per l'analisi.

- **Esecuzione delle Query:**  
  Il codice esegue diverse query per identificare specifici profili di rischio, ad esempio:
  - `alto_rischio(Età, Sesso)`
  - `ipertensione_colesterolo(Età, Sesso)`
  - `ecg_anomalo(Età, Sesso)`
  - `angina_sforzo(Età, Sesso)`
  - `profilo_alto_rischio(Età, Sesso)`

- **Output:**  
  I risultati delle query vengono formattati e stampati in modo chiaro, con l'uso di emoji per rendere la lettura più intuitiva:
  - 🔴 Pazienti ad ALTO RISCHIO
  - 🟠 Pazienti con IPERTENSIONE o COLESTEROLO ALTO
  - 🔵 Pazienti con ECG ANOMALO
  - ⚠️ Pazienti con ANGINA DA SFORZO
  - 🔥 Pazienti con PROFILO AD ALTO RISCHIO

---

## Esecuzione del Codice

Per eseguire correttamente questo progetto, seguire i seguenti passaggi:

1. **Installare le Dipendenze:**  
   Assicurarsi di avere installato le librerie richieste:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pyswip
