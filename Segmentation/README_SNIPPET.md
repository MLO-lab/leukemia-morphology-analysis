
# Segmentation Pipeline (Refactored, v3)

**Highlights**
- Process any subset of channels from {**BF**, **Nucleus**, **DF**}; do **not** treat absent ones as errors.
- Support **additional channels** (e.g., `Actin`, `Ch2`) that are segmented like BF (cyto3).
- **DF special rule**: if BF exists, DF reuses BF masks when possible; if DF alone, segment it like BF.
- **Dry run** produces exactly one file: `dry_run_report_<k>.txt` (no images, no other logs).
- **Idempotent**: skip sets that are fully processed for the roles present in that set.
- **Per-run manifest CSV** (`manifest_<k>.csv`) summarizing what happened per image (non-dry-run).

**Quick start**
1. Copy `segment_cells_refactored.py` and `resources_example.txt`.
2. Edit paths and tags; optionally set extra roles:
   ```ini
   additional_roles=Ch2,Actin
   Ch2_tags=Ch2
   Actin_tags=Actin
   ```
3. Run a **planning pass**:
   ```bash
   # dry run: reports planned outputs, sets with no matching channels, and already-processed sets
   python segment_cells_refactored.py resources.txt
   ```
   This writes only: `dry_run_report_0.txt`.
4. Set `dry_run=false` to run segmentation. The run writes versioned logs and `manifest_0.csv`.

**Manifest columns**
- `run_idx, rel_dir, role, image_file, action, picked_diam, area_used, note`

**Smoke test (dry-run)**
- See `smoke_test_dry_run.py` for an automated check that creates a tiny fake dataset and runs a dry pass.
