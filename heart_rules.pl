% File heart_rules.pl
:- consult('heart_kb.pl').

% Disabilita i warning per variabili singole
    :- style_check(-singleton).

    % ======== Regole per l'analisi delle malattie cardiache ========

% Pazienti a rischio elevato se hanno almeno due fattori critici
alto_rischio(Età, Sesso) :-
    heart_patient(Età, Sesso, TipoDolore, PressioneSistolica, Colesterolo, Glicemia, _, _, _, VecchioPicco, _, _, _, _),
    (TipoDolore > 1; PressioneSistolica > 140; Colesterolo > 250; Glicemia = 1; VecchioPicco > 2.0).

% Pazienti con ipertensione e/o colesterolo alto
ipertensione_colesterolo(Età, Sesso) :-
    heart_patient(Età, Sesso, _, PressioneSistolica, Colesterolo, _, _, _, _, _, _, _, _, _),
    (PressioneSistolica > 140; Colesterolo > 250).

% Pazienti con ECG anomalo (qualsiasi valore superiore a 0)
ecg_anomalo(Età, Sesso) :-
    heart_patient(Età, Sesso, _, _, _, _, ECG, _, _, _, _, _, _, _),
    ECG > 0.

% Pazienti con angina da sforzo (valore 1)
angina_sforzo(Età, Sesso) :-
    heart_patient(Età, Sesso, _, _, _, _, _, _, Angina, _, _, _, _, _),
    Angina = 1.

% Pazienti a basso rischio se non hanno fattori critici
basso_rischio(Età, Sesso) :-
    heart_patient(Età, Sesso, TipoDolore, PressioneSistolica, Colesterolo, Glicemia, _, FrequenzaMax, _, VecchioPicco, _, _, _, Malattia),
    TipoDolore = 0,
    PressioneSistolica =< 120,
    Colesterolo =< 200,
    Glicemia = 0,
    FrequenzaMax > 140,
    VecchioPicco =< 1.0,
    Malattia = 0.

% Profilo alto rischio: combina più criteri di rischio
profilo_alto_rischio(Età, Sesso) :-
    alto_rischio(Età, Sesso);
    ecg_anomalo(Età, Sesso);
    angina_sforzo(Età, Sesso).