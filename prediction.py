
# predict_with_renaming.py
# ------------------------
# Usage:
#   python predict_with_renaming.py --features /path/to/skimage_features.csv \
#       --model /path/to/xgb_checkpoint.pkl \
#       [--feature-list /path/to/feature_names.txt] \
#       [--output-dir /path/to/out] \
#       [--id-cols Image_Type,Image_Name]
#
# Notes:
# - Accepts models saved as sklearn XGBClassifier/Regressor (joblib/pickle), or xgboost.Booster (.json/.ubj/.model).
# - If the model does not carry feature names, pass --feature-list: a text file with one feature per line,
#   or a CSV header line (comma-separated).
# - The script always creates a *normalized* copy of your features with new notation (_BF/_Nucleus/_DF).
# - For prediction, it maps back to the model's expected names and order.
import argparse, os, sys, json, re
from pathlib import Path as _Path
from typing import List, Optional
import numpy as np
import pandas as pd

def read_table_auto(path: str) -> pd.DataFrame:
    # Try pandas' automatic separator inference
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback to comma
        return pd.read_csv(path)

def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- name mapping ---
_dot_suffix_pat = re.compile(r"^(.*)\.(\d+)$")

def old_to_new(col: str) -> str:
    """Map old names to new (_BF/_Nucleus/_DF)."""
    m = _dot_suffix_pat.match(col)
    if m:
        base, num = m.group(1), m.group(2)
        if num == "1":
            return f"{base}_Nucleus"
        elif num == "2":
            return f"{base}_DF"
        else:
            # Unknown numeric suffix; keep as-is
            return col
    # Already new-style?
    if col.endswith(("_BF","_Nucleus","_DF")):
        return col
    # Otherwise assume BF
    return f"{col}_BF"

def new_to_old(col: str) -> str:
    """Map new names to old (base, base.1, base.2)."""
    if col.endswith("_BF"):
        return col[:-3]
    if col.endswith("_Nucleus"):
        return col[:-8] + ".1"
    if col.endswith("_DF"):
        return col[:-3] + ".2"
    # If doesn't match, return as-is
    return col

def normalize_feature_names(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """Return a copy of df with feature columns renamed to new scheme; ID columns untouched."""
    cols = []
    for c in df.columns:
        if c in id_cols:
            cols.append(c)
        else:
            cols.append(old_to_new(c))
    out = df.copy()
    out.columns = cols
    return out

def extract_model_feature_names(model) -> Optional[List[str]]:
    names = None
    # sklearn API
    for attr in ("feature_names_in_",):
        if hasattr(model, attr):
            try:
                arr = getattr(model, attr)
                names = [str(x) for x in list(arr)]
                if names:
                    return names
            except Exception:
                pass
    # xgboost sklearn wrapper
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster is not None and getattr(booster, "feature_names", None):
                names = list(booster.feature_names)
                if names:
                    return names
        except Exception:
            pass
    # raw Booster
    if getattr(model, "feature_names", None):
        try:
            names = list(model.feature_names)
            if names:
                return names
        except Exception:
            pass
    return None

def load_feature_list(path: str) -> List[str]:
    text = open(path, "r", encoding="utf-8").read().strip()
    if "\n" in text and "," not in text.splitlines()[0]:
        # One per line
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    # else assume CSV header or comma-separated
    first = text.splitlines()[0]
    return [x.strip() for x in first.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Predict with XGBoost using renamed features and aligned column order.")
    ap.add_argument("--features", required=True, help="Path to skimage features CSV")
    ap.add_argument("--model", required=True, help="Path to XGBoost checkpoint (.pkl/.joblib or Booster .json/.ubj/.model)")
    ap.add_argument("--feature-list", default=None, help="Optional path to text/CSV listing expected feature names (in model's training order)")
    ap.add_argument("--output-dir", default=None, help="Where to write outputs; default is the features file's directory")
    ap.add_argument("--id-cols", default="Image_Type,Image_Name", help="Comma-separated ID columns to carry through")
    args = ap.parse_args()

    feats_path = args.features
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(feats_path))
    os.makedirs(out_dir, exist_ok=True)

    # Load features
    df_in = read_table_auto(feats_path)
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    for c in id_cols:
        if c not in df_in.columns:
            # tolerate missing ID columns
            pass

    # Normalize to new naming
    df_norm = normalize_feature_names(df_in, id_cols=id_cols)
    norm_csv = os.path.join(out_dir, _Path(feats_path).stem + "_normalized.csv")
    df_norm.to_csv(norm_csv, index=False)
    print(f"[INFO] Wrote normalized (new-notation) features: {norm_csv}")

    # Load model
    model = None
    model_path = args.model
    load_err = None
    try:
        import joblib
        model = joblib.load(model_path)
    except Exception as e:
        load_err = e

    booster = None
    if model is None:
        # Try Booster
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(model_path)
            model = booster
        except Exception as e2:
            print(f"[ERROR] Failed to load model as joblib ({load_err}) and as Booster ({e2})")
            sys.exit(1)

    # Determine expected feature order (model training order)
    expected_names = extract_model_feature_names(model)
    if expected_names is None:
        if args.feature_list:
            expected_names = load_feature_list(args.feature_list)
            print(f"[WARN] Model did not contain feature names; using list from --feature-list ({len(expected_names)} names).")
        else:
            print("[ERROR] Could not obtain feature names from model. Provide --feature-list pointing to names in training order.")
            sys.exit(2)

    # Build prediction matrix in expected order.
    # The model was trained on *old* notation; map old->new when selecting from df_norm.
    new_cols_set = set(df_norm.columns)
    missing_model_cols = []
    aligned_cols = []
    for old_name in expected_names:
        new_name = old_to_new(old_name)  # map model's old name to our normalized new name
        if new_name in new_cols_set:
            aligned_cols.append(new_name)
        else:
            missing_model_cols.append((old_name, new_name))
            aligned_cols.append(None)

    if missing_model_cols:
        print(f"[WARN] {len(missing_model_cols)} model features not found in input; filling with 0.0")
        for old_name, new_name in missing_model_cols[:10]:
            print(f"       missing: model '{old_name}' (expecting column '{new_name}')")

    # Build X matrix
    X_parts = []
    for col in aligned_cols:
        if col is None:
            X_parts.append(np.zeros((len(df_norm), 1), dtype=float))
        else:
            X_parts.append(pd.to_numeric(df_norm[col], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
    X = np.hstack(X_parts) if X_parts else np.zeros((len(df_norm), 0))

    # Predict
    proba = None
    pred = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            pred = getattr(model, "predict")(X)
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=expected_names)
            out = model.predict(dmat)
            if isinstance(out, list):
                out = np.array(out)
            # Heuristic: if 1D, treat as probabilities for positive class; if 2D, assume class probs
            if out.ndim == 1:
                proba = np.vstack([1 - out, out]).T
                pred = (out >= 0.5).astype(int)
            else:
                proba = out
                pred = np.argmax(out, axis=1)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        sys.exit(3)

    # Build JSON-friendly outputs
    # IDs per row
    ids = []
    for i in range(len(df_norm)):
        item = {}
        for c in id_cols:
            if c in df_norm.columns:
                item[c] = df_norm.loc[i, c]
        ids.append(item)

    # Probabilities JSON
    proba_list = []
    if proba is not None:
        for i, row in enumerate(np.asarray(proba).tolist()):
            rec = {"index": i, "proba": row}
            rec.update(ids[i])
            proba_list.append(rec)

    # Predictions JSON
    pred_list = []
    if pred is not None:
        for i, y in enumerate(np.asarray(pred).tolist()):
            rec = {"index": i, "pred": y}
            rec.update(ids[i])
            pred_list.append(rec)

    base = _Path(feats_path).stem
    proba_json = os.path.join(out_dir, base + "_proba.json")
    pred_json = os.path.join(out_dir, base + "_pred.json")
    write_json(proba_json, proba_list)
    write_json(pred_json, pred_list)
    print(f"[OK] Wrote: {proba_json}")
    print(f"[OK] Wrote: {pred_json}")

if __name__ == "__main__":
    main()
