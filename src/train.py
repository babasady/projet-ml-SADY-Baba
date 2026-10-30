"""Entraînement du détecteur d'attaques réseau (IDS) - Projet 2 ML.

Usage (depuis la racine du dépôt) :
    python src/train.py

Étapes : exploration -> prétraitement (Pipeline) -> comparaison de plusieurs
modèles sur un jeu de validation -> choix du meilleur (F1 classe « attaque »)
-> ré-entraînement sur toutes les données -> sauvegarde dans models/model.joblib.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "flux_train.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
TARGET = "attaque"
SEED = 42

CAT_COLS = ["protocol"]
# Variables à forte asymétrie : on applique log1p pour les modèles linéaires.
SKEWED_COLS = ["duration", "dst_port", "src_bytes", "dst_bytes",
               "nb_packets", "nb_failed_logins", "nb_conn_same_host"]
OTHER_NUM_COLS = ["syn_rate"]


def explore(df: pd.DataFrame) -> None:
    """Exploration rapide : taille, valeurs manquantes, répartition de la cible."""
    print("=== EXPLORATION ===")
    print(f"Taille : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    missing = df.isna().sum()
    print("\nValeurs manquantes :")
    print(missing[missing > 0].to_string())
    print("\nRépartition de 'attaque' :")
    print(df[TARGET].value_counts().to_string())
    print((df[TARGET].value_counts(normalize=True) * 100).round(1).to_string())
    print()


def make_preprocessor(log_transform: bool) -> ColumnTransformer:
    """Imputation + mise à l'échelle des numériques, one-hot de 'protocol'."""
    skewed_steps = [("imputer", SimpleImputer(strategy="median"))]
    if log_transform:
        skewed_steps.append(
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    skewed_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("skewed", Pipeline(skewed_steps), SKEWED_COLS),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), OTHER_NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def build_models() -> dict:
    return {
        "Régression logistique": Pipeline([
            ("prep", make_preprocessor(log_transform=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED))]),
        "Arbre de décision": Pipeline([
            ("prep", make_preprocessor(log_transform=False)),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                           class_weight="balanced",
                                           random_state=SEED))]),
        "Forêt aléatoire": Pipeline([
            ("prep", make_preprocessor(log_transform=False)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=SEED))]),
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocessor(log_transform=False)),
            ("clf", GradientBoostingClassifier(random_state=SEED))]),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    explore(df)

    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    print("=== COMPARAISON DES MODÈLES (validation 25 %) ===")
    for name, pipe in build_models().items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
        results[name] = {
            "f1": f1_score(y_val, pred),
            "recall": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred),
            "cv_f1": cv_f1,
        }
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"\n--- {name} ---")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print(f"Matrice de confusion [[TN FP] [FN TP]] : [[{tn} {fp}] [{fn} {tp}]]")
        print(f"Faux négatifs (attaques ratées) : {fn} / {fn + tp}")
        print(f"F1 attaque (validation) = {results[name]['f1']:.3f} | "
              f"F1 moyen validation croisée 5 plis = {cv_f1:.3f}")

    summary = pd.DataFrame(results).T.round(3)
    print("\n=== RÉCAPITULATIF ===")
    print(summary.to_string())

    # Choix du meilleur modèle : F1 moyen en validation croisée (plus stable
    # qu'un seul découpage), départagé par le F1 sur le jeu de validation.
    best_name = summary.sort_values(["cv_f1", "f1"], ascending=False).index[0]
    print(f"\nModèle retenu : {best_name}")

    # Ré-entraînement sur TOUTES les données puis sauvegarde.
    best = build_models()[best_name]
    best.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
