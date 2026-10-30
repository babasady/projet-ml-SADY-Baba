"""Prédiction avec le modèle entraîné (Pipeline complet : prétraitement + modèle)."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_model = None


def _load_model():
    """Charge le modèle une seule fois."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(df: pd.DataFrame) -> list:
    """Prédit 0 (normal) ou 1 (attaque) pour chaque flux.

    df : mêmes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0/1, une valeur par ligne.
    """
    model = _load_model()
    if "attaque" in df.columns:
        df = df.drop(columns="attaque")
    return [int(p) for p in model.predict(df)]


if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "flux_train.csv")
    preds = predict(data.drop(columns="attaque"))
    print(f"{len(preds)} prédictions, {sum(preds)} attaques détectées")
