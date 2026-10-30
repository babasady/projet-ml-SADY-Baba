# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Niveau** : L3 · **Travail** : individuel · **Langage** : Python (pandas, scikit-learn)

Un système de détection d'intrusion (IDS) doit décider, pour chaque flux réseau, s'il est
**normal (0)** ou une **attaque (1)** (scan de ports, force brute SSH/Telnet, DoS).

## Structure du dépôt
```
projet-ml-SADY-Baba/
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py          # exploration, comparaison de 4 modèles, sauvegarde du meilleur
├── src/predict.py        # fonction predict(df) -> list de 0/1
├── models/model.joblib   # Pipeline complet (prétraitement + modèle)
└── reports/train_output.txt   # sortie complète de l'entraînement
```

## Utilisation
```bash
pip install -r requirements.txt
python src/train.py      # réentraîne et écrase models/model.joblib
python src/predict.py    # démo sur le jeu d'entraînement
```
```python
import pandas as pd
from src.predict import predict
df = pd.read_csv("data/flux_train.csv").drop(columns="attaque")
predict(df)   # [1, 0, 1, 0, ...]
```
> Le modèle a été sérialisé avec les versions de `requirements.txt` (scikit-learn 1.8.0).
> Utiliser les mêmes versions pour charger `model.joblib`, ou relancer `python src/train.py`.

## Démarche
1. **Exploration** : 3000 flux, 9 variables, 90 valeurs manquantes (uniquement `duration`),
   28 % d'attaques (833) contre 72 % de flux normaux (2167) → classes déséquilibrées, on suit le **F1 de la classe « attaque »**.
2. **Prétraitement** (dans un `Pipeline` / `ColumnTransformer`) : imputation par la médiane,
   `StandardScaler` sur les numériques, `OneHotEncoder` sur `protocol`. Pour la régression logistique,
   transformation `log1p` des variables très asymétriques (octets, durée, compteurs).
3. **Modèles comparés** : régression logistique, arbre de décision (profondeur 6), forêt aléatoire (300 arbres),
   Gradient Boosting. `class_weight="balanced"` pour les trois premiers.
4. **Évaluation** : découpage stratifié 75 % / 25 % (graine 42) + validation croisée stratifiée à 5 plis (F1).
5. **Modèle final** : ré-entraîné sur les 3000 flux et sauvegardé dans `models/model.joblib`.

---
## Résultats

**Nom / Prénom :** SADY Baba

Jeu de validation : 750 flux (208 attaques).

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 moyen (CV 5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,962 | 0,971 | 0,953 | 0,952 |
| Arbre de décision | 0,966 | 0,966 | 0,966 | 0,939 |
| **Forêt aléatoire** | **0,973** | 0,971 | 0,976 | **0,959** |
| Gradient Boosting | 0,973 | 0,966 | 0,980 | 0,953 |

**Matrices de confusion** (validation, lignes = réel, colonnes = prédit ; [[VN, FP], [FN, VP]]) :

| Modèle | Matrice | Faux négatifs |
|---|---|---|
| Régression logistique | [[532, 10], [6, 202]] | 6 |
| Arbre de décision | [[535, 7], [7, 201]] | 7 |
| Forêt aléatoire | [[537, 5], [6, 202]] | 6 |
| Gradient Boosting | [[538, 4], [7, 201]] | 7 |

**Modèle retenu et pourquoi :** la **forêt aléatoire**. Elle a le meilleur F1 moyen en validation croisée
(0,959), ce qui est plus fiable qu'un seul découpage, et un F1 de validation de 0,973 (à égalité avec le Gradient Boosting)
avec le meilleur rappel du lot (0,971). Les modèles sont très proches (écarts de 1 à 3 points de F1) : la régression
logistique, plus simple et interprétable, reste un très bon compromis. La forêt est robuste aux valeurs extrêmes,
gère les relations non linéaires entre variables et généralise mieux que l'arbre seul (moins de variance).
Variables les plus importantes : `dst_bytes` (0,38), `nb_conn_same_host` (0,23), `syn_rate` (0,13), `nb_failed_logins` (0,11).
Cela correspond à la logique réseau : une attaque reçoit peu d'octets en retour, multiplie les connexions vers la même
machine, avec beaucoup de SYN et d'échecs d'authentification.

**Analyse des erreurs (faux négatifs / faux positifs) :**
- **Faux négatifs : 6 attaques ratées sur 208 (2,9 %)**. Ce sont des attaques **discrètes** : peu de connexions vers
  la même machine (3 à 9), `syn_rate` faible (0,04–0,32), aucun échec d'authentification, et un `dst_bytes` de type
  trafic légitime (plusieurs milliers d'octets en retour). Elles ressemblent à du trafic normal (ports 80, 443, 8080, 22).
- **Faux positifs : 5 flux normaux sur 542 (0,9 %)** classés à tort en attaque : flux courts avec forte activité
  (`nb_conn_same_host` élevé, `syn_rate` élevé ou nombreux échecs de login), proches d'un scan ou d'une force brute.
- **Pourquoi le faux négatif est le pire cas en sécurité :** une attaque non détectée passe sous le radar, personne
  n'est alerté et l'attaquant peut continuer (compromission, vol de données, indisponibilité du service) sans réaction.
  Un faux positif coûte seulement une alerte à vérifier par un analyste. On privilégie donc un **rappel élevé**.
  Pour le réduire davantage, on pourrait abaisser le seuil de décision (`predict_proba` < 0,5), au prix de plus de faux positifs.
