# Predizione delle Malattie Cardiache e Analisi Basata su Conoscenza

Questo progetto in Python esegue un'analisi approfondita sui dati relativi alle malattie cardiache e utilizza modelli di apprendimento supervisionato per la predizione. In particolare, vengono impiegati i modelli **Decision Tree** e **Random Forest** per la classificazione, e viene integrato un sistema basato su regole tramite Prolog per ottenere ulteriori informazioni (ad esempio, l'identificazione dei pazienti ad alto rischio). Inoltre, è implementato un algoritmo di ricerca in ampiezza (BFS) per trovare pazienti simili.

---

## Contenuto del Progetto

### 1. Data Exploration e Visualizzazione
- **Caricamento e analisi del dataset:**  
  Il dataset viene caricato da un file CSV e vengono mostrate informazioni come il numero di righe e colonne, le statistiche descrittive, il controllo dei valori mancanti e le prime righe del dataset.

- **Visualizzazione:**  
  Vengono generate diverse visualizzazioni, tra cui:
  - Istogrammi per ogni variabile.
  - Grafico a barre per la distribuzione della variabile target.
  - Heatmap della matrice di correlazione per evidenziare le relazioni tra le variabili.

### 2. Pre-elaborazione dei Dati
- **Trasformazione delle variabili categoriche:**  
  Le variabili categoriche vengono trasformate in variabili dummy (one-hot encoding) utilizzando `pd.get_dummies()`.

- **Standardizzazione:**  
  Le variabili numeriche vengono standardizzate (media 0 e deviazione standard 1) tramite `StandardScaler`.

- **Suddivisione del dataset:**  
  I dati vengono suddivisi in variabili indipendenti (X) e variabile target (y), e successivamente divisi in set di training (80%) e test (20%) con `train_test_split`.

### 3. Costruzione e Valutazione dei Modelli di Classificazione

#### Decision Tree
- **Validazione Incrociata:**  
  Viene testata la performance di alberi decisionali con profondità variabile (da 1 a 10) utilizzando la validazione incrociata (10-fold).  
- **Addestramento:**  
  Viene addestrato un modello di Decision Tree con `max_depth=3` sui dati di training.
- **Valutazione:**  
  Le prestazioni del modello vengono valutate sul set di test calcolando le seguenti metriche:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC  
  Queste metriche sono calcolate e stampate tramite la funzione `stampa_metriche`.

  ![Decision Tree](image/decisiontree.png)

#### Random Forest
- **Validazione Incrociata:**  
  Viene eseguita la validazione incrociata (5-fold) per modelli Random Forest variando il numero di stimatori (da 10 a 100, a passi di 10).
- **Addestramento:**  
  Viene addestrato un modello Random Forest con `n_estimators=90` sui dati di training.
- **Valutazione:**  
  Le metriche di valutazione (come per il Decision Tree) vengono calcolate sul set di test.
  
  ![Random Forest](image/randomforest.png)

### 4. Integrazione con Prolog per l'Analisi Basata su Regole
- **Consultazione e Query:**  
  Il programma utilizza il modulo `pyswip` per interfacciarsi con Prolog. Viene consultato il file `heart_rules.pl` (che include anche `heart_kb.pl`) e vengono eseguite diverse query per ottenere:
  - Pazienti ad alto rischio (`alto_rischio`)
  - Pazienti con ipertensione o colesterolo alto (`ipertensione_colesterolo`)
  - Pazienti con anomalie all'ECG (`ecg_anomalo`)
  - Pazienti con angina da sforzo (`angina_sforzo`)
  - Pazienti con profilo ad alto rischio (`profilo_alto_rischio`)

  I risultati vengono troncati per visualizzare al massimo 3 elementi per ciascuna query.

### 5. Algoritmo BFS per la Ricerca di Pazienti Simili
- **Implementazione della BFS:**  
  Viene implementato un algoritmo di ricerca in ampiezza (BFS) per trovare pazienti simili basandosi su una condizione definita nella query Prolog `heart_patient`.  
- **Funzionamento:**  
  - Si parte da un paziente iniziale (definito da età e sesso).
  - La ricerca si espande in ampiezza fino a una profondità massima (default 3).
  - Si evitano duplicazioni tenendo traccia dei pazienti già visitati.
  - Se il numero di risultati supera 3, viene aggiunto un segnaposto per indicare il troncamento.

---

## Requisiti

Il progetto richiede l'installazione delle seguenti librerie Python:
- numpy
- pandas
- matplotlib
- seaborn
- pyswip
- scikit-learn

Inoltre, è necessario avere i seguenti file nella struttura del progetto:
- `heart.csv` (dataset)
- `heart_rules.pl` (file Prolog contenente le regole, che include anche `heart_kb.pl`)
- La cartella `image` contenente:
  - `decisiontree.png`
  - `randomforest.png`

---

## Istruzioni per l'Esecuzione

1. **Installazione delle Dipendenze:**  
   Installa le librerie richieste, ad esempio con:
   ```bash
   pip install numpy pandas matplotlib seaborn pyswip scikit-learn
