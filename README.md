
# Modélisation Épidémiologique par le Modèle SIRD  

### **Description du Projet**

Ce projet a été réalisé dans le cadre du cours d'Analyse Numérique de la deuxième année du cycle préparatoire à l'ENSAM Casablanca. L'objectif principal est d'étudier et de modéliser la dynamique de propagation d'un virus au sein d'une population en utilisant des méthodes numériques avancées.

Le projet se concentre sur l'application du modèle SIRD (Susceptible, Infected, Recovered, Dead) pour simuler et analyser la propagation du COVID-19 (SARS-CoV-2). Nous utilisons des techniques d'interpolation, de résolution d'équations non linéaires, de dérivation numérique, d'intégration numérique et de résolution d'équations différentielles pour modéliser et prédire l'évolution de l'épidémie.


### **Structure du Projet**

```
├── README.md
├── requirements.txt
├── setup.py
├── data/
├── notebooks/
│   ├── 1_Interpolation.ipynb
│   ├── 2_Equations_Non_Lineaires.ipynb
│   ├── 3_Derivation_Numerique.ipynb
│   ├── 4_Integration_Numerique.ipynb
│   ├── 5_Equations_Differentielles.ipynb
│   └── Data_Exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── estimateur_parametres.py
│   │   ├── derivation/
│   │   │   └── methodes.py
│   │   ├── equations_differentielles/
│   │   │   ├── __init__.py
│   │   │   ├── simulateur_sird.py
│   │   │   └── solveur.py
│   │   ├── integration/
│   │   │   └── methodes.py
│   │   ├── interpolation/
│   │   │   └── methodes.py
│   │   └── resolution_eq_non_lineaire/
│   │       ├── __init__.py
│   │       └── solveur.py
│   └── data/
│       ├── __init__.py
│       ├── cleaner.py
│       ├── fetcher.py
│       └── validator.py
```

### **Description des Répertoires et Fichiers**
- **notebooks/**: Contient les notebooks Jupyter qui détaillent chaque étape du projet.
- **src/**: Contient les modules Python utilisés dans les notebooks.
    - **analysis/**: Modules pour l'analyse des données et la résolution des équations.
    - **data/**: Modules pour le traitement des données (nettoyage, téléchargement, validation).
### **Gestion des Données**
Les données COVID-19 sont automatiquement :
- Téléchargées depuis [Our World in Data](https://covid.ourworldindata.org/)
- Nettoyées et prétraitées via `src/data/cleaner.py`
- Validées avec `src/data/validator.py`

Stockage :
- Données brutes : `data/raw/`
- Données traitées : `data/processed/`


### **Détails Supplémentaires**

- **Modularité**: Le projet est structuré de manière modulaire pour faciliter la maintenance et l'évolutivité. Les scripts de traitement des données, les modules d'analyse et les notebooks sont séparés pour une meilleure organisation.
  
- **Réutilisabilité**: Les modules dans `src/` peuvent être importés et réutilisés dans différents notebooks ou scripts, ce qui permet de centraliser le code et de réduire les redondances.

---

### **Installation et Configuration**

### Prérequis
- Python 3.10+
- Git

### Configuration de l'environnement

#### Avec venv (méthode native) :
```bash
git clone https://github.com/votre_utilisateur/virus-spread-model.git
cd virus-spread-model

# Création et activation de l'environnement
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate.bat  # Windows

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . jupyter
```

#### Avec Conda :
```bash
git clone https://github.com/votre_utilisateur/virus-spread-model.git
cd virus-spread-model

# Création et activation de l'environnement
conda create -n virus_env python=3.13
conda activate virus_env

# Installer Jupyter
conda install jupyter -c conda-forge

# Installation des dépendances
pip install -r requirements.txt
pip install -e . 
```

### Lancement des notebooks
```bash
jupyter notebook  # Ou jupyter lab
```

---

Merci d'avoir consulté ce README. Nous espérons que ce projet vous sera utile pour comprendre la modélisation de la propagation des virus et l'importance de l'analyse numérique dans ce domaine.
