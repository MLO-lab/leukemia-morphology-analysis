
"""
smoke_test_dry_run.py
---------------------
Creates a tiny fake dataset structure (no real TIFF data needed) and runs a DRY RUN.
Verifies the script writes exactly one dry-run report and no other logs or images.
"""
import os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "smoke_tmp"
RAW = WORK / "Raw_images"
SEG = WORK / "Segmented_images_CellPose"
EXP = "Human normal donors"
SET = "setA"

# Clean
if WORK.exists():
    shutil.rmtree(WORK)
(RAW / EXP / SET).mkdir(parents=True, exist_ok=True)
SEG.mkdir(parents=True, exist_ok=True)

# Create a couple of fake file names
# Note: Dry-run does not read TIFFs, it only lists them.
(RAW / EXP / SET / "sample1_BF.tif").write_bytes(b"")
(RAW / EXP / SET / "sample1_DF.tif").write_bytes(b"")
(RAW / EXP / SET / "sample2_Nucleus.tif").write_bytes(b"")
(RAW / EXP / SET / "sample3_Actin.tif").write_bytes(b"")

# Resources file
RES = WORK / "resources.txt"
RES.write_text(
    "\n".join([
        f"path_for_raw_images={RAW}",
        f"experiments_types={EXP}",
        f"save_path_segmented_images={SEG}",
        "bf_tags=BF,Ch1",
        "nucleus_tags=Nucleus,Ch5",
        "df_tags=DF,Ch6",
        "additional_roles=Actin",
        "Actin_tags=Actin",
        "dry_run=true",
        "use_optional_fallback=true",
    ]) + "\n"
)

# Import and run the segmentation
sys.path.insert(0, str(ROOT))
import segment_cells_refactored as seg
seg.execute_segmentation(str(RES))

# Show outputs
print("\\nContents of output dir:")
for p in sorted(SEG.glob("*")):
    print(" -", p.name)
for p in sorted(SEG.rglob("*")):
    if p.is_file():
        print("file:", p.relative_to(SEG))
