# Predizione delle Malattie Cardiache e Analisi Basata su Conoscenza

Questo progetto in Python esegue un'analisi approfondita sui dati relativi alle malattie cardiache e utilizza modelli di apprendimento supervisionato (Decision Tree e Random Forest) per la predizione. Inoltre, integra un sistema basato su regole tramite Prolog per estrarre ulteriori informazioni (ad esempio, l'identificazione dei pazienti ad alto rischio) e per eseguire una ricerca in ampiezza (BFS) per trovare pazienti simili.

---

## Contenuto del Progetto

- **Data Exploration e Visualizzazione**  
  Il programma carica un dataset in formato CSV, esegue analisi esplorative (forma, statistiche descrittive, controllo dei valori mancanti) e crea varie visualizzazioni (istogrammi, grafico a barre, heatmap della correlazione) per comprendere la distribuzione dei dati.

- **Pre-elaborazione dei Dati**  
  Le variabili categoriche vengono trasformate in variabili dummy (one-hot encoding) e le variabili numeriche vengono standardizzate per avere media 0 e deviazione standard 1. Successivamente, il dataset viene diviso in variabili indipendenti (X) e variabile target (y).

- **Divisione in Set di Training e Test**  
  Utilizzando `train_test_split` il dataset viene suddiviso in un set di training (80%) e un set di test (20%) per addestrare e valutare i modelli.

- **Costruzione e Valutazione dei Modelli di Classificazione**  
  - **Decision Tree:**  
    Viene testata la performance di alberi decisionali con profondità variabile (da 1 a 10) tramite validazione incrociata a 10 fold. I risultati vengono visualizzati graficamente e, infine, viene addestrato un modello con `max_depth=3` sui dati di training.  
  - **Random Forest:**  
    Viene eseguita la validazione incrociata a 5 fold per modelli Random Forest con numero di stimatori variabile (da 10 a 100, a passi di 10). Dopo aver visualizzato le performance, viene addestrato un modello con `n_estimators=90`.

  Per entrambi i modelli, le prestazioni vengono valutate su un set di test calcolando le seguenti metriche:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **ROC-AUC**

  Una funzione `stampa_metriche` centralizza il calcolo e la stampa di queste metriche.

- **Integrazione con Prolog per l'Analisi Basata su Regole**  
  Il programma utilizza il modulo `pyswip` per interfacciarsi con Prolog. Vengono consultati i file `heart_rules.pl` (che include anche `heart_kb.pl`) e vengono eseguite diverse query per ottenere informazioni relative a:
  - Pazienti ad alto rischio (`alto_rischio`)
  - Pazienti con ipertensione o colesterolo alto (`ipertensione_colesterolo`)
  - Pazienti con anomalie all'ECG (`ecg_anomalo`)
  - Pazienti con angina da sforzo (`angina_sforzo`)
  - Pazienti con profilo ad alto rischio (`profilo_alto_rischio`)

  I risultati vengono troncati a un massimo di 3 output per una visualizzazione più compatta.

- **Algoritmo BFS per la Ricerca di Pazienti Simili**  
  Viene implementato un algoritmo di Breadth-First Search (BFS) per trovare pazienti simili basandosi su una condizione definita nella query Prolog `heart_patient`.  
  L'algoritmo:
  - Parte da un paziente iniziale (definito da età e sesso).
  - Espande la ricerca in ampiezza fino a una profondità massima (default 3).
  - Evita di visitare ripetutamente lo stesso paziente.
  - Restituisce un elenco di pazienti simili (con un segnaposto se il numero di risultati supera 3).

---

## Requisiti

Il progetto richiede l'installazione delle seguenti librerie Python:
- numpy
- pandas
- matplotlib
- seaborn
- pyswip
- scikit-learn

È inoltre necessario avere i file:
- `heart.csv` (il dataset)
- `heart_rules.pl` (file Prolog contenente le regole, che include anche `heart_kb.pl`)

---

## Istruzioni per l'Esecuzione

1. **Installazione delle Dipendenze:**  
   Assicurati di avere installato i pacchetti necessari. Puoi installarli utilizzando pip:
   ```bash
   pip install numpy pandas matplotlib seaborn pyswip scikit-learn
