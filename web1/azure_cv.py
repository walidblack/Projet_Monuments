"""
azure_cv.py : classification d'une image avec Azure Custom Vision.

Configuration (sans écrire les clés dans le code) :
  - soit le fichier web/azure_config.json
  - soit les variables d'environnement
    AZURE_PREDICTION_KEY, AZURE_ENDPOINT, AZURE_PROJECT_ID, AZURE_PUBLISHED_NAME
"""
import json
import os
from pathlib import Path

import requests

CONFIG_FILE = Path(__file__).resolve().parent / "azure_config.json"
ENV_NAMES = {
    "prediction_key": "AZURE_PREDICTION_KEY",
    "endpoint": "AZURE_ENDPOINT",
    "project_id": "AZURE_PROJECT_ID",
    "published_name": "AZURE_PUBLISHED_NAME",
}
TIMEOUT = 20  # secondes


class AzureNotConfigured(Exception):
    pass


def load_config():
    file_cfg = {}
    if CONFIG_FILE.exists():
        file_cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    cfg = {k: (file_cfg.get(k) or os.environ.get(env, "")).strip()
           for k, env in ENV_NAMES.items()}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise AzureNotConfigured(
            "Azure n'est pas configuré : complétez azure_config.json "
            f"(champs vides : {', '.join(missing)}).")
    return cfg


def classify(image_path):
    """Retourne [(étiquette, probabilité en %)] triée de la plus haute à la plus basse."""
    cfg = load_config()
    url = (f"{cfg['endpoint'].rstrip('/')}/customvision/v3.0/Prediction/"
           f"{cfg['project_id']}/classify/iterations/{cfg['published_name']}/image")
    headers = {"Prediction-Key": cfg["prediction_key"],
               "Content-Type": "application/octet-stream"}

    with open(image_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f.read(), timeout=TIMEOUT)
    response.raise_for_status()

    preds = [(p["tagName"], round(p["probability"] * 100, 2))
             for p in response.json()["predictions"]]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds


def classify_safe(image_path, top=5):
    """Ne lève jamais d'exception : retourne {'predictions': [...], 'error': message ou None}."""
    out = {"predictions": [], "error": None}
    try:
        out["predictions"] = classify(image_path)[:top]
    except AzureNotConfigured as e:
        out["error"] = str(e)
    except requests.exceptions.Timeout:
        out["error"] = "Azure ne répond pas (délai dépassé)."
    except requests.exceptions.ConnectionError:
        out["error"] = "Connexion à Azure impossible : vérifiez internet et l'endpoint."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if code in (401, 403):
            out["error"] = "Clé Azure refusée : vérifiez prediction_key."
        elif code == 404:
            out["error"] = "Projet ou itération introuvable : vérifiez project_id et published_name."
        elif code == 400:
            out["error"] = "Azure a refusé l'image (taille ou format)."
        else:
            out["error"] = f"Erreur Azure (code {code})."
    except Exception as e:
        out["error"] = f"Erreur Azure : {e}"
    return out
