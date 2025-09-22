# Image_Features_Extraction_cif.py  (v4 — TXT resources, custom export folder, _BF/_Nucleus/_DF suffixes)
import os, sys
from typing import List, Dict
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage import measure

# ----------------- config loader (TXT) -----------------

def _parse_list(val: str) -> List[str]:
    if val is None: return []
    parts = []
    for chunk in str(val).replace(";", ",").split(","):
        p = chunk.strip()
        if p and p not in parts:
            parts.append(p)
    return parts

def _load_config(txt_path: str) -> dict:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Resources file not found: {txt_path}")
    txt = open(txt_path, "r", encoding="utf-8").read().replace("\r\n","\n").replace("\\n","\n")
    raw = {}
    for line in txt.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "#" in s: s = s.split("#",1)[0].strip()
        if "=" in s:
            k, v = s.split("=",1)
        elif ":" in s:
            k, v = s.split(":",1)
        else:
            continue
        raw[k.strip().lower()] = v.strip().strip('"').strip("'")

    cfg = {}
    cfg["images_path"] = raw.get("images_path","").strip()
    cfg["features_export_path"] = raw.get("features_export_path","").strip()
    cfg["experiments_types"] = _parse_list(raw.get("experiments_types",""))
    if not cfg["images_path"] or not cfg["features_export_path"] or not cfg["experiments_types"]:
        raise ValueError("Required keys: images_path, features_export_path, experiments_types")

    # role tags
    cfg["role_tags"] = {
        "BF": _parse_list(raw.get("bf_tags","BF")),
        "Nucleus": _parse_list(raw.get("nucleus_tags","Nucleus")),
        "DF": _parse_list(raw.get("df_tags","DF")),
    }
    add_roles = _parse_list(raw.get("additional_roles",""))
    add_tags = {}
    for r in add_roles:
        tags = raw.get(f"{r.lower()}_tags", raw.get(f"{r}_tags", r))
        add_tags[r] = _parse_list(tags)
    cfg["additional_roles"] = add_roles
    cfg["additional_tags"] = add_tags

    # properties
    default_props = ["area",
        "area_bbox",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "bbox",
        "convex_area",
        "equivalent_diameter",
        "filled_area",
        "major_axis_length",
        "minor_axis_length",
        "weighted_centroid",
        "centroid",
        "centroid_local",
        "centroid_weighted",
        "centroid_weighted_local",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        #"image",
        #"image_convex",
        #"image_filled",
        #"image_intensity",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        #"label",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "moments_weighted",
        "moments_weighted_central",
        "moments_weighted_hu",
        "moments_weighted_normalized",
        #"num_pixels",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        #"slice",
        "solidity"]
    props = _parse_list(raw.get("properties_list",""))
    cfg["properties_list"] = (props if props else default_props)
    return cfg

# ----------------- helpers -----------------

def all_roles(cfg) -> List[str]:
    roles = ["BF","Nucleus","DF"]
    for r in cfg["additional_roles"]:
        if r not in roles:
            roles.append(r)
    return roles

def role_tags(cfg, role: str) -> List[str]:
    if role in ("BF","DF","Nucleus"):
        return cfg["role_tags"].get(role, [role])
    return cfg["additional_tags"].get(role, [role])

def union_tags(cfg) -> List[str]:
    tags = set()
    for r in all_roles(cfg):
        for t in role_tags(cfg, r):
            tags.add(t)
    return sorted(tags, key=lambda t: (-len(t), t.lower()))

def define_image_dirs(images_root: str, experiments: List[str]) -> Dict[str, List[str]]:
    exp_to_dirs = {}
    for exp in experiments:
        exp_root = os.path.join(images_root, exp)
        if not os.path.isdir(exp_root): continue
        dirs = []
        for root, _d, files in os.walk(exp_root):
            if any(f.lower().endswith(('.tif','.tiff')) for f in files):
                dirs.append(root)
        exp_to_dirs[exp] = dirs
    return exp_to_dirs

def first_subfolder_label(abs_dir: str, exp_root: str) -> str:
    rel = abs_dir.replace(exp_root, "").lstrip(os.sep)
    parts = [p for p in rel.split(os.sep) if p]
    return parts[0] if parts else "root"

def base_key_for_file(fname: str, all_tags: List[str]) -> str:
    low = fname.lower()
    for tag in all_tags:
        i = low.find(tag.lower())
        if i >= 0:
            return fname[:i] + "{CH}" + fname[i+len(tag):]
    return fname

# ----------------- main -----------------

def extract_skimage_features(resources_txt_path: str):
    cfg = _load_config(resources_txt_path)
    os.makedirs(cfg["features_export_path"], exist_ok=True)

    tags_all = union_tags(cfg)
    exp_to_dirs = define_image_dirs(cfg["images_path"], cfg["experiments_types"])

    for exp, dirs in exp_to_dirs.items():
        exp_root = os.path.join(cfg["images_path"], exp)
        label_to_dirs = {}
        for d in dirs:
            label = first_subfolder_label(d, exp_root)
            label_to_dirs.setdefault(label, []).append(d)

        for label, label_dirs in sorted(label_to_dirs.items()):
            rows = []
            for abs_dir in label_dirs:
                files = [f for f in os.listdir(abs_dir) if f.lower().endswith(('.tif','.tiff'))]
                groups = {}
                for f in files:
                    k = base_key_for_file(f, tags_all)
                    groups.setdefault(k, []).append(f)

                for base_key, flist in groups.items():
                    # map role -> representative file
                    role_to_file = {}
                    for f in flist:
                        low = f.lower()
                        for r in all_roles(cfg):
                            if any(t.lower() in low for t in role_tags(cfg, r)):
                                role_to_file.setdefault(r, f)
                                break

                    # compute props per present role
                    role_dfs = []
                    for r, f in sorted(role_to_file.items()):
                        img = skio.imread(os.path.join(abs_dir, f))
                        lbl = (img > 0).astype('int')
                        props = measure.regionprops_table(lbl, intensity_image=img, properties=cfg["properties_list"])
                        df = pd.DataFrame(props).add_suffix(f"_{r}")
                        role_dfs.append(df)

                    if not role_dfs:
                        continue

                    merged = None
                    for df in role_dfs:
                        merged = df if merged is None else pd.concat([merged, df], axis=1)
                    merged.insert(0, "Image_Name", base_key.replace("{CH}", "SET"))
                    merged.insert(0, "Image_Type", f"{exp}/{label}")
                    rows.append(merged)

            if not rows:
                continue

            out_df = pd.concat(rows, axis=0, ignore_index=True)
            out_name = f"skimage_{exp}_{label}.csv"
            out_path = os.path.join(cfg["features_export_path"], out_name)
            out_df.to_csv(out_path, index=False)
            print("Wrote:", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Image_Features_Extraction_cif.py <resources_txt_path>")
        raise SystemExit(1)
    extract_skimage_features(sys.argv[1])
