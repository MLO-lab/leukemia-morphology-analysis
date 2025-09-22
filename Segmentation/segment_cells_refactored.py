"""
segment_cells_refactored.py  (v4.3)
-----------------------------------
- Detect channels per set from configured tags (BF/DF/Nucleus + additional roles).
- Build per-base "sets" (e.g., 11_Ch1*, 11_Ch5*, 11_Ch6*). If any *required* core channel is
  missing for a set, SKIP THE ENTIRE SET and record it to channels_unavailability_<k>.txt (or dry-run).
- Process ONLY complete sets; plan/segment standardized output names by replacing the matched tag
  with: BF, DF, Nucleus, or the additional role name.
- DF reuses BF mask if BF output exists for the matching base.
- Dry run writes a single dry_run_report_<k>.txt (includes skipped sets & planned outputs).
- Logs:
    * fallback_segmentation_warning_<k>.txt   -> only images where fallback WAS used (and succeeded)
    * segmentation_failed_<k>.txt             -> only images where Cellpose failed AND fallback failed
    * channels_unavailability_<k>.txt         -> per-set lines for missing required core channels
    * successful_segmentations_<k>.txt        -> every output created (success/reuse/fallback success)
    * planned_outputs_<k>.txt, already_processed_sets_<k>.txt, area_flagged_..._<k>.txt, manifest_<k>.csv
"""

from __future__ import annotations

import os, re, glob, csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import imageio
from skimage import io as skio
from skimage import measure

try:
    from cellpose import models
except Exception:
    models = None  # allow import on systems without Cellpose (e.g., dry-run environments)

# ---------------------------
# Configuration I/O (TXT)
# ---------------------------

@dataclass
class Config:
    path_for_raw_images: str
    experiments_types: List[str]
    save_path_segmented_images: str

    channel_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "BF": ["BF", "Ch1"],
        "Nucleus": ["Nucleus", "Ch5"],
        "DF": ["DF", "Ch6"],
    })

    additional_roles: List[str] = field(default_factory=list)
    additional_aliases: Dict[str, List[str]] = field(default_factory=dict)

    gpu: bool = True
    dry_run: bool = False
    use_optional_fallback: bool = True

    model_cyto: str = "cyto3"
    model_nuclei: str = "nuclei"
    diameters_df: List[int] = field(default_factory=lambda: [16, 15, 14, 13, 12])
    diameters_bf: List[int] = field(default_factory=lambda: [16, 15, 14, 13, 12])
    diameters_nucleus: List[int] = field(default_factory=lambda: [13, 12, 11, 10])
    diameters_additional: List[int] = field(default_factory=lambda: [16, 15, 14, 13, 12])

    axis_ratio_thresh_df: float = 1.6
    axis_ratio_thresh_nucleus: float = 1.7
    area_min_df: int = 100
    area_min_nucleus: int = 80
    area_min_additional: int = 100

    # Log bases (actual files get suffixed with _<k>.txt / .csv)
    log_corrupted: str = "fallback_segmentation_warning"
    log_error: str = "segmentation_failed"
    log_area_flagged: str = "area_flagged_corrupted_segmentations_paths_New"
    log_channels_unavailability: str = "channels_unavailability"
    log_already_processed: str = "already_processed_sets"
    log_planned_outputs: str = "planned_outputs"
    log_dry_report: str = "dry_run_report"
    log_success: str = "successful_segmentations"

    # Which core channels were explicitly declared in resources (BF/Nucleus/DF)
    declared_core_roles: List[str] = field(default_factory=list)

    manifest_base: str = "manifest"


def _parse_list(val: str) -> List[str]:
    parts = []
    for chunk in str(val).replace(";", ",").split(","):
        p = chunk.strip()
        if p and p not in parts:
            parts.append(p)
    return parts


def _parse_int_list(val: str) -> List[int]:
    return [int(x.strip()) for x in _parse_list(val)]


def _parse_bool(val: str, default: bool=False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1","true","t","yes","y")


def load_config_from_txt(txt_path: str) -> Config:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Resources file not found: {txt_path}")

    # Read and normalize newlines; tolerate literal "\n"
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = raw_text.replace("\r\n", "\n")
    if "\\n" in raw_text:
        raw_text = raw_text.replace("\\n", "\n")

    raw: Dict[str, str] = {}
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        k = k.strip().lower()
        v = v.strip().strip('"').strip("'")
        if k:
            raw[k] = v

    path_for_raw_images = raw.get("path_for_raw_images", "").strip()
    experiments_types = _parse_list(raw.get("experiments_types", ""))
    save_path_segmented_images = raw.get("save_path_segmented_images", "").strip()
    if not path_for_raw_images or not experiments_types or not save_path_segmented_images:
        raise ValueError(
            "Required keys: path_for_raw_images, experiments_types, save_path_segmented_images "
            f"(parsed keys: {sorted(raw.keys())}) from file: {txt_path}"
        )

    channel_aliases = {
        "BF": _parse_list(raw.get("bf_tags", "BF,Ch1")),
        "Nucleus": _parse_list(raw.get("nucleus_tags", "Nucleus,Ch5")),
        "DF": _parse_list(raw.get("df_tags", "DF,Ch6")),
    }

    additional_roles = _parse_list(raw.get("additional_roles", ""))
    additional_aliases: Dict[str, List[str]] = {}
    for role in additional_roles:
        rr = role.lower()
        tags = raw.get(f"{rr}_tags", raw.get(f"{role}_tags", role))
        additional_aliases[role] = _parse_list(tags)

    cfg = Config(
        path_for_raw_images=path_for_raw_images,
        experiments_types=experiments_types,
        save_path_segmented_images=save_path_segmented_images,
        channel_aliases=channel_aliases,
        additional_roles=additional_roles,
        additional_aliases=additional_aliases,
        gpu=_parse_bool(raw.get("gpu","true"), True),
        dry_run=_parse_bool(raw.get("dry_run","false"), False),
        use_optional_fallback=_parse_bool(raw.get("use_optional_fallback","true"), True),
        model_cyto=raw.get("model_cyto","cyto3"),
        model_nuclei=raw.get("model_nuclei","nuclei"),
        diameters_df=_parse_int_list(raw.get("diameters_df","16,15,14,13,12")),
        diameters_bf=_parse_int_list(raw.get("diameters_bf","16,15,14,13,12")),
        diameters_nucleus=_parse_int_list(raw.get("diameters_nucleus","13,12,11,10")),
        diameters_additional=_parse_int_list(raw.get("diameters_additional","16,15,14,13,12")),
        axis_ratio_thresh_df=float(raw.get("axis_ratio_thresh_df","1.6")),
        axis_ratio_thresh_nucleus=float(raw.get("axis_ratio_thresh_nucleus","1.7")),
        area_min_df=int(raw.get("area_min_df","100")),
        area_min_nucleus=int(raw.get("area_min_nucleus","80")),
        area_min_additional=int(raw.get("area_min_additional","100")),
        log_corrupted=raw.get("log_corrupted","fallback_segmentation_warning"),
        log_error=raw.get("log_error","segmentation_failed"),
        log_area_flagged=raw.get("log_area_flagged","area_flagged_corrupted_segmentations_paths_New"),
        log_channels_unavailability=raw.get("log_channels_unavailability","channels_unavailability"),
        log_already_processed=raw.get("log_already_processed","already_processed_sets"),
        log_planned_outputs=raw.get("log_planned_outputs","planned_outputs"),
        log_dry_report=raw.get("log_dry_report","dry_run_report"),
        log_success=raw.get("log_success","successful_segmentations"),
        manifest_base=raw.get("manifest_base","manifest"),
    )

    # Determine required core channels based on which *_tags were actually present in resources
    declared_core = []
    if "bf_tags" in raw and raw.get("bf_tags", "").strip():
        declared_core.append("BF")
    if "nucleus_tags" in raw and raw.get("nucleus_tags", "").strip():
        declared_core.append("Nucleus")
    if "df_tags" in raw and raw.get("df_tags", "").strip():
        declared_core.append("DF")
    if not declared_core:
        # If user didn't specify any core keys, default to all three
        declared_core = ["BF", "Nucleus", "DF"]
    cfg.declared_core_roles = declared_core

    return cfg


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def _write_line(p: str, s: str) -> None:
    _ensure_dir(p)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(s + "\n")


def _all_run_indices(out_root: str, bases: List[str], also_csv: Optional[str]=None) -> List[int]:
    idxs = []
    for b in bases:
        for fp in glob.glob(os.path.join(out_root, f"{b}_*.txt")):
            m = re.search(r"_(\d+)\.txt$", fp)
            if m:
                idxs.append(int(m.group(1)))
    if also_csv:
        for fp in glob.glob(os.path.join(out_root, f"{also_csv}_*.csv")):
            m = re.search(r"_(\d+)\.csv$", fp)
            if m:
                idxs.append(int(m.group(1)))
    return sorted(set(idxs))


def allocate_run_index(cfg: Config) -> int:
    bases = [
        cfg.log_corrupted, cfg.log_error, cfg.log_area_flagged,
        cfg.log_channels_unavailability, cfg.log_already_processed,
        cfg.log_planned_outputs, cfg.log_dry_report, cfg.log_success
    ]
    used = _all_run_indices(cfg.save_path_segmented_images, bases, also_csv=cfg.manifest_base)
    if not used:
        return 0
    return max(used) + 1


def versioned_path(cfg: Config, base: str, run_idx: int, ext: str="txt") -> str:
    return os.path.join(cfg.save_path_segmented_images, f"{base}_{run_idx}.{ext}")


def define_image_dirs(path_for_raw_images: str, experiments: Iterable[str]) -> List[str]:
    image_dirs: List[str] = []
    for exp in experiments:
        exp_path = os.path.join(path_for_raw_images, exp)
        if not os.path.isdir(exp_path):
            continue
        for root, _dirs, files in os.walk(exp_path):
            if any(fname.lower().endswith((".tif", ".tiff")) for fname in files):
                image_dirs.append(root)
    rel = [p.replace(path_for_raw_images, "").lstrip(os.sep) for p in image_dirs]
    return rel


def find_files_with_tags(folder: str, tag_list: List[str]) -> List[str]:
    files = []
    tags_low = [t.lower() for t in tag_list]
    for fname in os.listdir(folder):
        low = fname.lower()
        if not low.endswith((".tif", ".tiff")):
            continue
        for tag in tags_low:
            if tag in low:
                files.append(fname)
                break
    return sorted(list(set(files)))


def standardize_output_name(image_file: str, role: str, tags: List[str]) -> str:
    """
    Replace the matched tag in image_file with the standardized role name (case-insensitive).
    We choose the longest matching tag to avoid partial collisions.
    """
    best_tag = None
    best_pos = -1
    lower_name = image_file.lower()
    for t in tags:
        pos = lower_name.find(t.lower())
        if pos >= 0:
            if best_tag is None or len(t) > len(best_tag):
                best_tag, best_pos = t, pos
    if best_tag is None:
        return image_file  # fallback: keep original name
    # Replace the span [best_pos, best_pos+len(tag)) with the role (preserve the rest)
    return image_file[:best_pos] + role + image_file[best_pos+len(best_tag):]


def save_images(out_root: str, rel_dir: str, out_name: str, image: np.ndarray, mask: np.ndarray) -> None:
    contours = measure.find_contours(mask, level=0.2)
    out_png_dir = os.path.join(out_root, "Png_images", rel_dir)
    out_tif_dir = os.path.join(out_root, "Tiff_images", rel_dir)
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_tif_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    for contour in contours:
        ax[0].plot(contour[:, 1], contour[:, 0], "r", linewidth=0.7)
    ax[0].set_axis_off()
    ax[1].imshow(image * mask, cmap="gray")
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_png_dir, out_name.replace(".tiff", ".png").replace(".tif", ".png")), dpi=200)
    plt.close(fig)

    imageio.imwrite(os.path.join(out_tif_dir, out_name), (image * mask).astype("uint16"), format="TIFF")


def _postfilter_mask_choose_largest_comp(masks: np.ndarray,
                                         is_nucleus: bool,
                                         axis_ratio_thresh_df: float,
                                         axis_ratio_thresh_nucleus: float,
                                         area_min_df: int,
                                         area_min_nucleus: int,
                                         area_min_additional: int,
                                         role: str) -> Tuple[bool, np.ndarray, int]:
    props = measure.regionprops_table(
        masks,
        properties=("axis_major_length", "axis_minor_length", "area"),
    )
    if len(props.get("area", [])) == 0:
        return False, np.zeros_like(masks, dtype=bool), -1

    df = pd.DataFrame(props)
    df["axis_ratio"] = df["axis_major_length"] / np.maximum(df["axis_minor_length"], 1e-6)

    if is_nucleus:
        axis_thr = axis_ratio_thresh_nucleus
        area_thr = area_min_nucleus
        min_area_limit = 30
    else:
        axis_thr = axis_ratio_thresh_df
        area_thr = area_min_additional if role not in ("BF", "DF") else area_min_df
        min_area_limit = 50

    ok_mask = None
    used_area = area_thr
    while True:
        subset = df[(df["axis_ratio"] < axis_thr) & (df["area"] > area_thr)]
        if len(subset) > 0:
            idx = subset["area"].idxmax()
            label_to_keep = int(idx) + 1
            ok_mask = (masks == label_to_keep)
            break
        area_thr -= 5
        if area_thr < min_area_limit:
            break

    if ok_mask is None:
        return False, np.zeros_like(masks, dtype=bool), area_thr
    return True, ok_mask.astype(bool), used_area


def bf_mask_from_bf_output_if_available(out_root: str, rel_dir: str, bf_out_name: str) -> Optional[np.ndarray]:
    out_tif_dir = os.path.join(out_root, "Tiff_images", rel_dir)
    cand_path = os.path.join(out_tif_dir, bf_out_name)
    if os.path.exists(cand_path):
        temp_image = skio.imread(cand_path)
        return (temp_image > 0).astype(np.uint8)
    return None


def run_cellpose(image: np.ndarray, role: str, cfg: Config) -> Tuple[bool, np.ndarray, Dict[str, float]]:
    if models is None:
        return False, np.zeros_like(image, dtype=np.uint8), {"picked_diam": -1}

    if role == "Nucleus":
        model = models.Cellpose(gpu=cfg.gpu, model_type=cfg.model_nuclei)
        diameters = cfg.diameters_nucleus
        is_nucleus = True
    elif role in ("BF", "DF") or role in cfg.additional_roles:
        model = models.Cellpose(gpu=cfg.gpu, model_type=cfg.model_cyto)
        if role == "BF":
            diameters = cfg.diameters_bf
        elif role == "DF":
            diameters = cfg.diameters_df
        else:
            diameters = cfg.diameters_additional
        is_nucleus = False
    else:
        model = models.Cellpose(gpu=cfg.gpu, model_type=cfg.model_cyto)
        diameters = cfg.diameters_additional
        is_nucleus = False

    tried = []
    masks = None
    type_labels = np.array([0])
    for diam in diameters:
        masks, flows, styles, diams = model.eval(image, diameter=diam, channels=[0, 0])
        type_labels, _ = np.unique(masks, return_counts=True)
        tried.append(diam)
        if len(type_labels) > 1:
            break

    if len(type_labels) <= 1:
        return False, np.zeros_like(image, dtype=np.uint8), {"picked_diam": -1}

    ok, best_mask, used_area = _postfilter_mask_choose_largest_comp(
        masks,
        is_nucleus=is_nucleus,
        axis_ratio_thresh_df=cfg.axis_ratio_thresh_df,
        axis_ratio_thresh_nucleus=cfg.axis_ratio_thresh_nucleus,
        area_min_df=cfg.area_min_df,
        area_min_nucleus=cfg.area_min_nucleus,
        area_min_additional=cfg.area_min_additional,
        role=role,
    )
    if not ok:
        return False, np.zeros_like(image, dtype=np.uint8), {"picked_diam": tried[-1] if tried else -1}

    meta = {"picked_diam": tried[-1] if tried else -1, "area_used": used_area}
    return True, best_mask.astype(np.uint8), meta


def all_known_tags(cfg: Config) -> List[str]:
    """All known tag tokens (BF/DF/Nucleus + additional roles), longest-first."""
    tags = []
    for lst in cfg.channel_aliases.values():
        tags += lst
    for role in cfg.additional_roles:
        tags += cfg.additional_aliases.get(role, [role])
    return sorted(set(tags), key=lambda t: (-len(t), t.lower()))


def base_key_for_file(fname: str, tags: List[str]) -> str:
    """
    Collapse the channel token in a name to a placeholder so
    11_Ch1_X.tif, 11_Ch5_X.tif, 11_Ch6_X.tif map to the same base key.
    """
    low = fname.lower()
    for tag in tags:
        i = low.find(tag.lower())
        if i >= 0:
            return fname[:i] + "{CH}" + fname[i+len(tag):]
    return fname


# ---------------------------
# Orchestrator
# ---------------------------

def execute_segmentation(resources_txt_path: str) -> None:
    cfg = load_config_from_txt(resources_txt_path)
    run_idx = allocate_run_index(cfg)

    os.makedirs(cfg.save_path_segmented_images, exist_ok=True)

    p_warn = versioned_path(cfg, cfg.log_corrupted, run_idx)       # fallback used (success)
    p_fail = versioned_path(cfg, cfg.log_error, run_idx)           # cellpose+fallback failed
    p_area = versioned_path(cfg, cfg.log_area_flagged, run_idx)    # area threshold relaxed
    p_chan = versioned_path(cfg, cfg.log_channels_unavailability, run_idx)  # missing required core channels
    p_already = versioned_path(cfg, cfg.log_already_processed, run_idx)
    p_plan = versioned_path(cfg, cfg.log_planned_outputs, run_idx)
    p_dry = versioned_path(cfg, cfg.log_dry_report, run_idx)
    p_manifest = versioned_path(cfg, cfg.manifest_base, run_idx, ext="csv")
    p_success = versioned_path(cfg, cfg.log_success, run_idx)

    # In dry-run, create a single report up-front
    if cfg.dry_run:
        _ensure_dir(p_dry)
        with open(p_dry, "w", encoding="utf-8", newline="\n") as _f:
            _f.write("DRY RUN REPORT\n")
            _f.write(f"resources: {resources_txt_path}\n")
            _f.write(f"run_idx: {run_idx}\n")

    image_dirs = define_image_dirs(cfg.path_for_raw_images, cfg.experiments_types)
    if cfg.dry_run and not image_dirs:
        _write_line(p_dry, "No image directories found for the configured experiments.")
        print(f"DRY RUN report written: {p_dry}")
        return

    def report_dry(line: str):
        _write_line(p_dry, line)

    def report_planned(line: str):
        if cfg.dry_run:
            report_dry(line)
        else:
            _write_line(p_plan, line)

    def write_manifest_header():
        _ensure_dir(p_manifest)
        with open(p_manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_idx","rel_dir","role","image_file_in","image_file_out","action","picked_diam","area_used","note"])

    def add_manifest(rel_dir: str, role: str, image_in: str, image_out: str, action: str, picked_diam: int=-1, area_used: int=-1, note: str=""):
        with open(p_manifest, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([run_idx, rel_dir, role, image_in, image_out, action, picked_diam, area_used, note])

    declared_roles = ["Nucleus", "BF", "DF"] + cfg.additional_roles

    if not cfg.dry_run:
        write_manifest_header()

    for rel_dir in image_dirs:
        abs_dir = os.path.join(cfg.path_for_raw_images, rel_dir)

        # 1) Discover files for all declared roles
        role_to_files: Dict[str, List[str]] = {}
        for role in declared_roles:
            tags = cfg.channel_aliases[role] if role in ("BF", "DF", "Nucleus") \
                   else cfg.additional_aliases.get(role, [role])
            files = find_files_with_tags(abs_dir, tags)
            role_to_files[role] = files

        # 2) Per-set completeness check (skip entire set if required core channels missing)
        union_tags = all_known_tags(cfg)
        # Build groups: base_key -> {role: [files]}
        groups: Dict[str, Dict[str, List[str]]] = {}
        for role, files in role_to_files.items():
            for f in files:
                key = base_key_for_file(f, union_tags)
                groups.setdefault(key, {}).setdefault(role, []).append(f)

        incomplete: List[Tuple[str, List[str]]] = []
        for key, rolemap in groups.items():
            if any(r in rolemap for r in cfg.declared_core_roles):
                missing = [r for r in cfg.declared_core_roles if r not in rolemap]
                if missing:
                    incomplete.append((key, missing))

        if incomplete:
            for key, missing in incomplete:
                line = f"{rel_dir} :: BASE={key} :: MISSING {','.join(missing)}"
                if cfg.dry_run:
                    report_dry(f"SKIP set (incomplete core): {line}")
                else:
                    _write_line(p_chan, line)
            # remove all files belonging to incomplete sets
            bad = {k for k, _ in incomplete}
            for r in role_to_files:
                role_to_files[r] = [f for f in role_to_files[r] if base_key_for_file(f, union_tags) not in bad]

        # 3) Present roles after filtering
        present_roles = [r for r, fs in role_to_files.items() if fs]
        if not present_roles:
            msg = f"{rel_dir} :: NO_MATCHING_CHANNELS (after completeness filter)"
            if cfg.dry_run:
                report_dry(msg)
            else:
                _write_line(p_chan, msg)
            continue

        # 4) Plan standardized outputs & check completeness for present roles
        out_tif_dir = os.path.join(cfg.save_path_segmented_images, "Tiff_images", rel_dir)
        planned_missing: List[Tuple[str, str, str]] = []  # (role, infile, outfile)
        all_present_completed = True
        for role in present_roles:
            tags = cfg.channel_aliases[role] if role in ("BF","DF","Nucleus") else cfg.additional_aliases.get(role, [role])
            for image_file in role_to_files[role]:
                out_name = standardize_output_name(image_file, role, tags)
                out_path = os.path.join(out_tif_dir, out_name)
                if not os.path.exists(out_path):
                    all_present_completed = False
                    planned_missing.append((role, image_file, out_name))

        if all_present_completed:
            msg = f"{rel_dir} :: ALREADY PROCESSED (complete for present roles)"
            if cfg.dry_run:
                report_dry(msg)
            else:
                _write_line(p_already, msg)
            continue

        report_planned(f"{rel_dir} :: PLANNED TO PROCESS {len(planned_missing)} files")
        for role, image_in, image_out in planned_missing:
            report_planned(f"  - {rel_dir}/{image_out} [{role}]  (from {image_in})")

        if cfg.dry_run:
            continue

        # 5) Processing order: Nucleus -> BF -> DF -> additional roles (present only)
        ordered_roles: List[str] = []
        if "Nucleus" in present_roles: ordered_roles.append("Nucleus")
        if "BF" in present_roles: ordered_roles.append("BF")
        if "DF" in present_roles: ordered_roles.append("DF")
        for r in cfg.additional_roles:
            if r in present_roles:
                ordered_roles.append(r)

        # 6) Process
        for role in ordered_roles:
            tags = cfg.channel_aliases[role] if role in ("BF","DF","Nucleus") else cfg.additional_aliases.get(role, [role])
            for image_in in role_to_files[role]:
                out_name = standardize_output_name(image_in, role, tags)
                out_tif = os.path.join(cfg.save_path_segmented_images, "Tiff_images", rel_dir, out_name)
                if os.path.exists(out_tif):
                    add_manifest(rel_dir, role, image_in, out_name, action="skip_exists")
                    continue

                abs_img_path = os.path.join(cfg.path_for_raw_images, rel_dir, image_in)
                image = skio.imread(abs_img_path)

                # DF reuses BF mask if BF output exists for corresponding base
                if role == "DF" and "BF" in present_roles:
                    bf_out_name = standardize_output_name(image_in, "BF", cfg.channel_aliases["BF"])
                    bfmask = bf_mask_from_bf_output_if_available(cfg.save_path_segmented_images, rel_dir, bf_out_name)
                    if bfmask is not None:
                        save_images(cfg.save_path_segmented_images, rel_dir, out_name, image, bfmask)
                        _write_line(p_success, os.path.join(rel_dir, out_name))
                        add_manifest(rel_dir, role, image_in, out_name, action="reuse_bf_mask", note="DF reused BF mask")
                        continue

                ok, mask, meta = run_cellpose(image, role, cfg)
                seg_key = os.path.join(rel_dir, out_name)

                if ok:
                    save_images(cfg.save_path_segmented_images, rel_dir, out_name, image, mask)
                    _write_line(p_success, seg_key)
                    add_manifest(rel_dir, role, image_in, out_name, action="processed",
                                 picked_diam=meta.get("picked_diam",-1), area_used=meta.get("area_used",-1))
                    thr_default = (
                        cfg.area_min_nucleus if role == "Nucleus"
                        else (cfg.area_min_df if role in ("BF","DF") else cfg.area_min_additional)
                    )
                    if "area_used" in meta and meta["area_used"] != thr_default:
                        _write_line(p_area, f"{seg_key}, AreaRelaxedTo={meta['area_used']}")
                else:
                    if cfg.use_optional_fallback:
                        # Try fallback using the original input name (fallback expects raw filename)
                        called = False
                        try:
                            func = None
                            # try import by module name
                            for modname in ("Segmenting_Image", "Segmenting_Image_Cell_Pose_15"):
                                try:
                                    mod = __import__(modname)
                                    if hasattr(mod, "Segmentation_Program"):
                                        func = getattr(mod, "Segmentation_Program")
                                        break
                                except Exception:
                                    continue
                            # try local file if module import failed
                            if func is None:
                                import importlib.util
                                for pth in (
                                    os.path.join(os.getcwd(), "Segmenting_Image_Cell_Pose_15.py"),
                                    os.path.join(os.getcwd(), "Segmenting_Image.py"),
                                ):
                                    if os.path.exists(pth):
                                        spec = importlib.util.spec_from_file_location("fallback_seg", pth)
                                        mod2 = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(mod2)  # type: ignore
                                        if hasattr(mod2, "Segmentation_Program"):
                                            func = getattr(mod2, "Segmentation_Program")
                                            break
                            if func is not None:
                                current_path = os.path.join(cfg.path_for_raw_images, rel_dir)
                                func(image_in, cfg.path_for_raw_images, cfg.save_path_segmented_images, current_path)
                                called = True
                        except Exception:
                            called = False

                        if called:
                            _write_line(p_warn, seg_key)     # fallback used (success)
                            _write_line(p_success, seg_key)  # also count as success
                            add_manifest(rel_dir, role, image_in, out_name, action="fallback_used", note="fallback segmentation")
                        else:
                            _write_line(p_fail, seg_key)     # failed after fallback
                            add_manifest(rel_dir, role, image_in, out_name, action="error", note="cellpose failed; fallback failed")
                    else:
                        add_manifest(rel_dir, role, image_in, out_name, action="not_processed_fallback_disabled", note="cellpose failed")

    if cfg.dry_run:
        print(f"DRY RUN report written: {p_dry}")
    else:
        print("Logs written:")
        for p in [p_warn, p_success, p_fail, p_area, p_chan, p_already, p_plan, p_manifest]:
            print(" -", p)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python segment_cells_refactored.py <resources_txt_path>")
        raise SystemExit(1)
    execute_segmentation(sys.argv[1])
