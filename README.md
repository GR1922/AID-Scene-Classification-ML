# AID-Scene-Classification-ML

Il progetto mira a classificare le immagini del dataset AID nelle 30 classi disponibili. Si adotta il Transfer Learning usando EfficientNet‑B0 come feature extractor, mentre la classificazione finale è affidata a modelli tradizionali: Logistic Regression, SVM (RBF e lineare) e kNN (k=5).

---

|              |                                           |
|--------------|-------------------------------------------|
| Descrizione  | Classificazione di scene aeree del dataset AID |
| Autore       | Gregory Riggi                             |
| Corso        | Machine Learning @ UniKore                |
| Licenza      | MIT                                       |

---

## Sommario
- [Introduzione](#introduzione)
- [Requisiti](#requisiti)
- [Struttura del codice](#struttura-del-codice)
- [Riproducibilità](#riproducibilità)
- [Risultati](#risultati)
- [Nota tecnica: bug di shuffle durante feature extraction](#Nota-tecnica:-bug-di-shuffle-durante-feature-extraction)
- [Licenza](#licenza)
  
---

## Introduzione
La classificazione di immagini aeree (Remote Sensing Scene Classification) è un problema rilevante in ambiti come monitoraggio ambientale, pianificazione urbana e analisi territoriale.  
Le immagini aeree presentano elevata variabilità in termini di scala, illuminazione, orientamento e complessità semantica.  

In questo progetto l’obiettivo è classificare le immagini del dataset **Aerial Image Dataset (AID)** in una delle **30 classi** disponibili.  
La pipeline utilizza **Transfer Learning**: una CNN pre-addestrata (**EfficientNet-B0**) viene usata come **feature extractor**, mentre la classificazione finale è eseguita tramite algoritmi tradizionali di Machine Learning (Scikit-learn).

---

## Requisiti
Il progetto si basa su **Python 3.11** ed è progettato per essere eseguito su **Google Colab**.

Le dipendenze sono elencate nel file `requirements.txt` e possono essere installate con:

```
pip install -r requirements.txt
```

---

## Struttura del codice
Il codice è strutturato come segue:

```
main_repository/
│
├── data_classes/
│   └── aid_dataset.py
│
├── model_classes/
│   └── feature_extractor.py
│
├── prepare.sh
├── requirements.txt
├── train.py
├── test.py
├── README.md
├── LICENSE
└── .gitignore
```
- `data_classes/` contiene le classi per la gestione del dataset.
- `model_classes/` contiene le classi relative al modello / feature extractor.
- `train.py` addestra il classificatore finale e salva gli artefatti.
- `test.py` carica gli artefatti e valuta il modello.
- `prepare.sh` installa i requisiti del progetto.

---

## Riproducibilità

L’idea principale è che il progetto possa essere riprodotto eseguendo:

1. Aprire Colab e clonare repo
   
`!git clone https://github.com/GR1922/AID-Scene-Classification-ML.git`

`%cd AID-Scene-Classification-ML`

`!bash prepare.sh`

3. Montare Drive e lanciare train/test
   
`from google.colab import drive`

`drive.mount('/content/drive')`

`!python test.py --dataset_dir /content/drive/MyDrive/AID`

`!python test.py --dataset_dir /content/drive/MyDrive/AID`

Download AID Dataset:

-> https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets?resource=download-directory

---

## Risultati

La seguente tabella riporta l’accuracy sul validation set (split stratificato 80/20):

| Modello             | Validation Accuracy |
| ------------------- | ------------------- |
| Logistic Regression | 0.9255              |
| SVM RBF             | 0.9210              |
| Linear SVM          | 0.9015              |
| kNN (k=5)           | 0.8545              |

---

## Nota tecnica: bug di shuffle durante feature extraction

Durante lo sviluppo è stato identificato un problema: l’estrazione delle feature era inizialmente effettuata su un dataset tf.data con .shuffle() attivo.

Questo comportava che:
- l’ordine delle feature estratte veniva mescolato
- le label rimanevano nell’ordine originale
- feature e label risultavano disallineate

La conseguenza era un’accuracy vicina al (≈5%).
La correzione è stata effettuata estraendo le feature con dataset senza shuffle, garantendo corrispondenza 1:1 tra feature e label.

## Licenza

Questo progetto è concesso in licenza secondo i termini della MIT License.

