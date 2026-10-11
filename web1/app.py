"""
app.py : page web Flask avec deux méthodes de reconnaissance de monuments :
  - YOLOv8   : détection, modèle local (best.pt)
  - Azure    : classification, service en ligne (Custom Vision)
Si une méthode échoue, l'autre s'affiche quand même.

Dossier web/ :
    app.py, azure_cv.py, azure_config.json, classes.json, templates/index.html

Lancer depuis la racine du projet (avec le .venv actif) :
    python web/app.py
puis ouvrir http://127.0.0.1:5000
"""
import json
import uuid
from pathlib import Path

import cv2
from flask import Flask, render_template, request
from PIL import Image, ImageOps
from ultralytics import YOLO

import azure_cv

BASE = Path(__file__).resolve().parent

# ----------------- PARAMETRES -----------------
MODEL_CANDIDATES = [
    BASE / "model" / "best.pt",
    BASE.parent / "runs" / "detect" / "train-3" / "weights" / "best.pt",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError("best.pt introuvable. Vérifiez runs/detect/train-3/weights/best.pt")

CLASSES_JSON = BASE / "classes.json"
MODEL_NAME = "YOLOv8n (best.pt)"
SHOW_ALL = True            # True = YOLO affiche toutes les classes, même très peu sûres
CONF = 0.001 if SHOW_ALL else 0.25
MIN_CONF_RELIABLE = 25     # en dessous (en %), on prévient que le résultat est peu fiable
AZURE_TOP = 5              # nombre de lignes affichées pour Azure
ALLOWED = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}
# ----------------------------------------------

UPLOAD_DIR = BASE / "static" / "uploads"
RESULT_DIR = BASE / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024      # 10 Mo

model = YOLO(str(MODEL_PATH))                             # chargé une seule fois

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    monuments = json.load(f)


def monument_name(class_id):
    """Nom du JSON ; à défaut, nom connu par le modèle."""
    return monuments.get(str(class_id), model.names[class_id].replace("_", " "))


def run_yolo(image_path, stem):
    out = {"best": None, "predictions": [], "message": None, "error": None,
           "image_file": None}
    try:
        result = model.predict(str(image_path), conf=CONF, verbose=False)[0]

        # image annotée : seulement la détection la plus sûre
        if len(result.boxes):
            annotated = result[int(result.boxes.conf.argmax())].plot()
        else:
            annotated = result.plot()
        cv2.imwrite(str(RESULT_DIR / f"{stem}.jpg"), annotated)
        out["image_file"] = f"results/{stem}.jpg"

        # une ligne par classe : sa meilleure confiance
        best_per_class = {}
        for b in result.boxes:
            cid, c = int(b.cls), float(b.conf) * 100
            if c > best_per_class.get(cid, -1):
                best_per_class[cid] = c
        preds = [(monument_name(cid), c, MODEL_NAME) for cid, c in best_per_class.items()]
        preds.sort(key=lambda p: p[1], reverse=True)
        out["predictions"] = preds

        if preds:
            out["best"] = preds[0]
            if preds[0][1] < MIN_CONF_RELIABLE:
                out["message"] = ("Résultat peu fiable : la confiance maximale est "
                                  f"inférieure à {MIN_CONF_RELIABLE} %.")
        else:
            out["message"] = "Aucun monument détecté."
    except Exception as e:
        out["error"] = f"Erreur YOLO : {e}"
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"error": None, "analysed": False, "yolo": None, "azure": None,
           "image_file": None, "min_conf": MIN_CONF_RELIABLE}

    if request.method == "GET":
        return render_template("index.html", **ctx)

    file = request.files.get("image")
    if file is None or file.filename == "":
        ctx["error"] = "Aucune image sélectionnée."
        return render_template("index.html", **ctx)

    if Path(file.filename).suffix.lower() not in ALLOWED:
        ctx["error"] = "Format non accepté. Utilisez JPG, JFIF, PNG, WEBP ou BMP."
        return render_template("index.html", **ctx)

    stem = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{stem}.jpg"

    # conversion en JPG (accepte .jfif, corrige l'orientation, réduit les grandes
    # images : Azure refuse les fichiers de plus de 4 Mo)
    try:
        with Image.open(file.stream) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1600, 1600))
            im.save(upload_path, "JPEG", quality=90)
    except Exception:
        ctx["error"] = "Fichier illisible : ce n'est pas une image valide."
        return render_template("index.html", **ctx)

    try:
        ctx["yolo"] = run_yolo(upload_path, stem)
        ctx["azure"] = azure_cv.classify_safe(str(upload_path), top=AZURE_TOP)
        ctx["image_file"] = ctx["yolo"]["image_file"]
        ctx["analysed"] = True
    finally:
        upload_path.unlink(missing_ok=True)

    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=False)
