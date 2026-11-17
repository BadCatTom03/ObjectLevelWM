"""Freeze the user-approved 6/2/2 per-category video split.

Rules (researcher confirmed 2026-11-11):
- Per category: 6 train / 2 val / 2 test clips, stratified so that val and
  test each contain at least one non-static camera clip.
- Camera per clip_index follows generate_semantic32.py: CAMERAS[(clip+cat)%5],
  i.e. {static, tracking, static, pan, static} cycled — so per category the
  non-static clips are determined by (clip_index + category_index) % 5.

Writes semantic32_v1/splits.json. Run-once: refuses to overwrite.
"""
import argparse
import json
import shutil
from pathlib import Path

from project_paths import DATA_ROOT

ROOT = DATA_ROOT.parent
DEST = DATA_ROOT / "splits.json"
CAMERAS = ["static", "tracking", "static", "pan", "static"]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate-existing", action="store_true",
                    help="Preserve old split as splits_legacy_24_4_4.json before applying 6/2/2")
    args = ap.parse_args()
    if DEST.exists() and not args.migrate_existing:
        raise SystemExit("splits.json already frozen; refusing to overwrite")
    if DEST.exists() and args.migrate_existing:
        legacy = DEST.with_name("splits_legacy_24_4_4.json")
        if not legacy.exists():
            shutil.copy2(DEST, legacy)
        old_gold = DEST.with_name("gold_legacy_24_4_4.json")
        current_gold = DEST.with_name("gold.json")
        if current_gold.exists() and not old_gold.exists():
            shutil.copy2(current_gold, old_gold)
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    cats = []
    for row in manifest:
        if row["category"] not in cats:
            cats.append(row["category"])
    assert len(cats) == 32
    # choose which 2 of the 10 clips per category go to val/test so each gets
    # >=1 non-static camera clip when possible.
    rows = []
    for row in manifest:
        ci = row["clip_index"]
        cam = CAMERAS[(ci + row["category_index"]) % 5]
        rows.append({**row, "camera": cam})
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    split_rows = []
    for cat, clips in by_cat.items():
        clips = sorted(clips, key=lambda r: r["clip_index"])
        nonstatic = [r for r in clips if r["camera"] != "static"]
        static = [r for r in clips if r["camera"] == "static"]
        # test: one non-static + one static; val: one non-static + one static
        test_pick = [nonstatic[0], static[0]] if nonstatic else clips[:2]
        remaining = [r for r in clips if r not in test_pick]
        rem_nonstatic = [r for r in remaining if r["camera"] != "static"]
        rem_static = [r for r in remaining if r["camera"] == "static"]
        val_pick = [rem_nonstatic[0], rem_static[0]] if rem_nonstatic else remaining[:2]
        train_pick = [r for r in clips if r not in test_pick + val_pick]
        for role, picks in (("train", train_pick), ("validation", val_pick), ("test", test_pick)):
            for r in picks:
                split_rows.append({
                    "id": r["id"], "category": cat,
                    "split": role, "camera": r["camera"],
                    "position": r["position_request"], "seed": r["seed"],
                })
    out = {
        "frozen": "2026-11-11",
        "policy": "all 32 categories represented in each split; source videos disjoint within each category; stratified by camera",
        "per_category_split": "6 train / 2 validation / 2 test",
        "category_holdout": False,
        "host_pool_rule": "test-split host videos disjoint from train patch sources; 3 unseen hosts per test target patch",
        "rows": split_rows,
    }
    DEST.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("frozen", len(split_rows), "rows; 32 categories x 6/2/2")


if __name__ == "__main__":
    main()
