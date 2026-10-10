"""
SafeTrace matching demo (ideathon prototype)

Shows two ideas on SAFE test photos:
  A) Perceptual hash: catches identical / near-identical copies.
  B) Face vector (embedding): catches edited copies where the face is visible.

Use only photos of consenting team members or a public face dataset.
Never use explicit images or real victim images.

Usage:
  python safetrace_matching_demo.py --photo me.jpg
  python safetrace_matching_demo.py --photo me.jpg --others other_people_folder
  python safetrace_matching_demo.py --photo me.jpg --hash-only     (skips face part)

Install (free):  pip install pillow imagehash deepface tf-keras numpy
"""
import argparse, csv, os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import imagehash

HASH_MATCH_MAX = 10        # max hash distance (out of 64) to call a match; tune on test data
FACE_MATCH_MAX_DIST = 0.30  # max cosine distance to call a face match; tune on test data


def make_variants(img, out_dir):
    """Create edited copies of the original photo and save them. Returns {name: path}."""
    os.makedirs(out_dir, exist_ok=True)
    w, h = img.size
    v = {}
    v["resized_50pct"] = img.resize((w // 2, h // 2))
    v["filtered_bw_contrast"] = ImageEnhance.Contrast(ImageOps.grayscale(img).convert("RGB")).enhance(1.6)
    t = img.copy(); d = ImageDraw.Draw(t); d.rectangle([0, h - h // 8, w, h], fill=(0, 0, 0)); d.text((10, h - h // 9), "SAMPLE WATERMARK", fill=(255, 255, 255))
    v["text_overlay"] = t
    v["crop_center_70pct"] = img.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
    v["crop_center_45pct"] = img.crop((int(w * .275), int(h * .275), int(w * .725), int(h * .725)))
    v["mirrored"] = ImageOps.mirror(img)
    paths = {}
    for name, im in v.items():
        p = os.path.join(out_dir, name + ".png"); im.save(p); paths[name] = p
    p = os.path.join(out_dir, "jpeg_heavy_compression.jpg"); img.save(p, quality=12); paths["jpeg_heavy_compression"] = p
    return paths


def hash_distance(path_a, path_b):
    return imagehash.phash(Image.open(path_a)) - imagehash.phash(Image.open(path_b))


def face_vector(path):
    """Returns a face embedding, or None if no face is found (face must be visible)."""
    from deepface import DeepFace
    try:
        r = DeepFace.represent(img_path=path, model_name="Facenet512",
                               detector_backend="opencv", enforce_detection=True)
        return np.array(r[0]["embedding"])
    except Exception:
        return None


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True, help="original safe photo (face clearly visible)")
    ap.add_argument("--others", help="folder of photos of DIFFERENT people (false-positive test)")
    ap.add_argument("--out", default="variants", help="folder for generated edited copies")
    ap.add_argument("--hash-only", action="store_true", help="skip the face-vector part")
    a = ap.parse_args()

    original = Image.open(a.photo).convert("RGB")
    items = [(k, p, "edited copy") for k, p in make_variants(original, a.out).items()]
    if a.others:
        for p in sorted(glob.glob(os.path.join(a.others, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.basename(p), p, "different person"))

    orig_vec = None if a.hash_only else face_vector(a.photo)
    if not a.hash_only and orig_vec is None:
        print("No face found in the original photo. Use a clear, front-facing photo."); return

    rows = []
    for name, path, kind in items:
        hd = hash_distance(a.photo, path)
        row = {"image": name, "type": kind, "hash_distance": hd, "hash_match": "YES" if hd <= HASH_MATCH_MAX else "no"}
        if not a.hash_only:
            vec = face_vector(path)
            if vec is None:
                row.update(face_distance="", face_match="no face found")
            else:
                fd = round(cosine_distance(orig_vec, vec), 3)
                row.update(face_distance=fd, face_match="YES" if fd <= FACE_MATCH_MAX_DIST else "no")
        rows.append(row)

    cols = ["image", "type", "hash_distance", "hash_match"] + ([] if a.hash_only else ["face_distance", "face_match"])
    print("\n" + " | ".join(c.ljust(24 if c == "image" else 14) for c in cols))
    print("-" * (26 + 16 * (len(cols) - 1)))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(24 if c == "image" else 14) for c in cols))

    with open("results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows([{c: r.get(c, "") for c in cols} for r in rows])

    edited = [r for r in rows if r["type"] == "edited copy"]; others = [r for r in rows if r["type"] != "edited copy"]
    print(f"\nHash thresholds: distance <= {HASH_MATCH_MAX}.  Face threshold: cosine distance <= {FACE_MATCH_MAX_DIST}.")
    print(f"Edited copies found by hash: {sum(r['hash_match']=='YES' for r in edited)}/{len(edited)}")
    if not a.hash_only:
        print(f"Edited copies found by face vector: {sum(r['face_match']=='YES' for r in edited)}/{len(edited)}")
    if others:
        print(f"Different people wrongly matched (hash): {sum(r['hash_match']=='YES' for r in others)}/{len(others)}")
        if not a.hash_only:
            print(f"Different people wrongly matched (face): {sum(r['face_match']=='YES' for r in others)}/{len(others)}")
    print("Saved: results.csv and the edited copies in the output folder.")


if __name__ == "__main__":
    main()
