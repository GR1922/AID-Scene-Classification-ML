# AID-Scene-Classification-ML

Il progetto mira a classificare le immagini del dataset AID nelle 30 classi disponibili. Si adotta il Transfer Learning usando EfficientNet‑B0 come feature extractor, mentre la classificazione finale è affidata a modelli tradizionali: Logistic Regression, SVM (RBF e lineare) e kNN (k=5).
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

