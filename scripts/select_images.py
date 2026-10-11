"""
select_images.py
Choisit, pour chaque classe :
  - 24 images dans originale_data/train/<classe>  -> datasets/train/images
  -  6 images dans originale_data/val/<classe>    -> datasets/val/images
Conditions : image lisible, taille suffisante, aucun doublon
(doublon exact OU image quasi identique), y compris entre train et val.

Lancer depuis la racine du projet :
    python scripts/select_images.py
"""
import hashlib
import random
import sys
from pathlib import Path
from shutil import copy2

from PIL import Image

# ----------------- PARAMETRES -----------------
SRC = Path("originale_data")
DST = Path("datasets")
N_PER_CLASS = {"train": 24, "val": 6}
MIN_SIDE = 400          # plus petit côté minimum (pixels) : "bonne qualité"
MIN_SIDE_FALLBACK = 250 # utilisé seulement s'il manque des images
MAX_RATIO = 3.0         # évite les panoramas extrêmes (largeur/hauteur)
USE_NEAR_DUP = False    # False = on vérifie seulement les copies EXACTES (md5)
NEAR_DUP_DIST = 1       # utilisé seulement si USE_NEAR_DUP = True
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ----------------------------------------------


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(im, size=8):
    """Empreinte visuelle : deux images quasi identiques ont des empreintes proches."""
    g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            i = r * (size + 1) + c
            bits = (bits << 1) | (1 if px[i] > px[i + 1] else 0)
    return bits


def read_info(path):
    """Retourne les infos de l'image, ou None si elle est corrompue."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            return {"path": path, "w": w, "h": h, "md5": md5_of(path), "dh": dhash(im)}
    except Exception:
        return None


def good_shape(info):
    ratio = max(info["w"], info["h"]) / max(1, min(info["w"], info["h"]))
    return ratio <= MAX_RATIO


def main():
    random.seed(SEED)

    # sécurité : ne pas écraser un travail déjà commencé
    for split in N_PER_CLASS:
        folder = DST / split / "images"
        if folder.exists() and any(folder.iterdir()):
            print(f"STOP : {folder} contient déjà des images.")
            print("Renomme l'ancien dossier (ex. datasets -> datasets_old) puis relance.")
            sys.exit(1)

    seen_md5 = set()
    seen_dh = []          # empreintes des images déjà choisies (train + val)
    report = []

    def is_duplicate(info):
        if info["md5"] in seen_md5:
            return True
        if not USE_NEAR_DUP:
            return False
        return any(bin(info["dh"] ^ d).count("1") <= NEAR_DUP_DIST for d in seen_dh)

    def register(info):
        seen_md5.add(info["md5"])
        seen_dh.append(info["dh"])

    for split, n in N_PER_CLASS.items():      # train d'abord, puis val
        out_img = DST / split / "images"
        out_lbl = DST / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        split_dir = SRC / split
        classes = sorted(p for p in split_dir.iterdir() if p.is_dir())

        for cdir in classes:
            files = [p for p in cdir.iterdir() if p.suffix.lower() in EXTS]
            random.shuffle(files)

            infos = [i for i in (read_info(p) for p in files) if i]
            n_corrupt = len(files) - len(infos)

            chosen = []

            # passe 1 : bonne taille
            for info in infos:
                if len(chosen) == n:
                    break
                if min(info["w"], info["h"]) >= MIN_SIDE and good_shape(info) \
                        and not is_duplicate(info):
                    chosen.append(info)
                    register(info)

            # passe 2 (si pas assez) : les plus grandes images restantes
            fallback_used = 0
            if len(chosen) < n:
                rest = [i for i in infos if i not in chosen
                        and min(i["w"], i["h"]) >= MIN_SIDE_FALLBACK]
                rest.sort(key=lambda i: min(i["w"], i["h"]), reverse=True)
                for info in rest:
                    if len(chosen) == n:
                        break
                    if not is_duplicate(info):
                        chosen.append(info)
                        register(info)
                        fallback_used += 1

            cname = cdir.name.replace(" ", "_")
            for info in chosen:
                p = info["path"]
                copy2(p, out_img / f"{cname}_{p.name}")

            report.append((split, cdir.name, len(chosen), n, fallback_used, n_corrupt))

    print(f"\n{'SPLIT':<6} {'CLASSE':<34} {'CHOISIES':<10} {'PETITES*':<9} {'CORROMPUES'}")
    for split, cname, got, n, fb, bad in report:
        flag = "" if got == n else f"  <-- MANQUE {n - got}"
        print(f"{split:<6} {cname:<34} {got}/{n:<7} {fb:<9} {bad}{flag}")
    print("\n* PETITES = images entre "
          f"{MIN_SIDE_FALLBACK} et {MIN_SIDE} px prises faute de mieux.")

    for split in N_PER_CLASS:
        total = len(list((DST / split / "images").iterdir()))
        print(f"{split}: {total} images copiées dans {DST / split / 'images'}")


if __name__ == "__main__":
    main()
