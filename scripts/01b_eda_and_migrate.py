#!/usr/bin/env python3
"""
Phase 0b: Comprehensive EDA + Normalized Schema Migration

This script:
1. Reads ALL raw source JSONs (output_YYYY.json) year by year
2. Validates every entry (schema, types, file existence)
3. Builds a single normalized SQLite database (data/metadata_v2.db)
4. Creates train/val/test splits (stored in DB, not JSON)
5. Generates a full EDA report (docs/eda_report_v2.md)

Usage:
    python scripts/01b_eda_and_migrate.py [--skip-image-check]
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# ─── Project root ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 5000  # SQLite insert batch size

RAW_DIR = Path(os.getenv(
    "RAW_DATASET_DIR",
    "/home/phuoc/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset"
))
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DB_PATH = DATA_DIR / "metadata_v2.db"


# ─── Logging (simple, no silent errors) ───────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "eda_migrate.log", mode="w"),
    ],
)
log = logging.getLogger("eda")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load & Validate Raw Source Data
# ═══════════════════════════════════════════════════════════════════════

def load_yearly_jsons(raw_dir: Path) -> Tuple[List[Dict], Dict]:
    """
    Load all output_YYYY.json files one at a time.
    Returns (entries, quality_report).
    """
    json_files = sorted(raw_dir.glob("output_*.json"))
    if not json_files:
        log.error(f"No output_*.json files found in {raw_dir}")
        sys.exit(1)

    log.info(f"Found {len(json_files)} yearly JSON files in {raw_dir}")

    all_entries: List[Dict] = []
    quality = {
        "total_raw": 0,
        "per_year": {},
        "vienna_as_string": 0,
        "vienna_as_list": 0,
        "vienna_none": 0,
        "text_present": 0,
        "text_null": 0,
        "missing_file_field": 0,
        "missing_year_field": 0,
        "duplicate_filenames": 0,
        "all_keys_seen": set(),
        "year_range": [],
    }
    seen_filenames: Set[str] = set()

    for jf in json_files:
        year_label = jf.stem.replace("output_", "")
        log.info(f"  Loading {jf.name} ...")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"  FAILED to parse {jf.name}: {e}")
            continue

        year_count = len(data)
        quality["per_year"][year_label] = year_count
        quality["total_raw"] += year_count

        for entry in data:
            quality["all_keys_seen"].update(entry.keys())

            # ── Validate filename ──
            fname = entry.get("file")
            if not fname:
                quality["missing_file_field"] += 1
                continue
            if fname in seen_filenames:
                quality["duplicate_filenames"] += 1
                continue
            seen_filenames.add(fname)

            # ── Normalize vienna_codes ──
            vc = entry.get("vienna_codes")
            if vc is None:
                quality["vienna_none"] += 1
                entry["vienna_codes"] = []
            elif isinstance(vc, str):
                quality["vienna_as_string"] += 1
                entry["vienna_codes"] = [vc] if vc.strip() else []
            elif isinstance(vc, list):
                quality["vienna_as_list"] += 1
            else:
                log.warning(f"  Unexpected vienna_codes type: {type(vc)} for {fname}")
                entry["vienna_codes"] = []

            # ── Validate year ──
            yr = entry.get("year")
            if yr is None:
                quality["missing_year_field"] += 1
            else:
                try:
                    entry["year"] = int(yr)
                except (ValueError, TypeError):
                    log.warning(f"  Non-integer year '{yr}' for {fname}")
                    entry["year"] = None

            # ── Validate text ──
            txt = entry.get("text")
            if txt and str(txt).strip():
                quality["text_present"] += 1
            else:
                quality["text_null"] += 1
                entry["text"] = None

            all_entries.append(entry)

    quality["unique_entries"] = len(all_entries)
    quality["all_keys_seen"] = sorted(quality["all_keys_seen"])
    quality["year_range"] = sorted(quality["per_year"].keys())

    log.info(f"Loaded {quality['unique_entries']} unique entries "
             f"({quality['total_raw']} raw, {quality['duplicate_filenames']} duplicates skipped)")
    return all_entries, quality


def check_image_existence(entries: List[Dict], images_dir: Path) -> Dict:
    """Check how many metadata entries have corresponding images on disk."""
    log.info(f"Checking image existence in {images_dir} ...")

    # Build set of actual files (lowercase for case-insensitive matching)
    actual_files: Set[str] = set()
    for f in os.scandir(images_dir):
        if f.is_file():
            actual_files.add(f.name.lower())

    log.info(f"  Found {len(actual_files)} image files on disk")

    matched = 0
    missing = 0
    case_mismatch = 0
    missing_samples: List[str] = []

    for entry in tqdm(entries, desc="Checking images", leave=False):
        fname = entry["file"]
        # The metadata has .JPG but disk has .jpg
        fname_lower = fname.lower()
        # Also try stem + .jpg (metadata may say .JPG but disk has .jpg)
        fname_stem_jpg = Path(fname).stem.lower() + ".jpg"

        if fname.lower() in actual_files:
            matched += 1
            if fname not in actual_files:
                case_mismatch += 1
        elif fname_stem_jpg in actual_files:
            matched += 1
            case_mismatch += 1
        else:
            missing += 1
            if len(missing_samples) < 10:
                missing_samples.append(fname)

    result = {
        "images_on_disk": len(actual_files),
        "metadata_matched": matched,
        "metadata_missing_image": missing,
        "case_mismatches": case_mismatch,
        "missing_samples": missing_samples,
    }

    log.info(f"  Matched: {matched}, Missing: {missing}, Case mismatches: {case_mismatch}")
    if missing_samples:
        log.warning(f"  Sample missing: {missing_samples[:5]}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Build Normalized SQLite Database
# ═══════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Core trademark metadata (1 row per image, single source of truth)
CREATE TABLE IF NOT EXISTS trademarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL UNIQUE,
    tm_text     TEXT,
    year        INTEGER,
    split       TEXT    CHECK(split IN ('train', 'val', 'test'))
);

-- Normalized Vienna codes (deduplicated)
CREATE TABLE IF NOT EXISTS vienna_codes (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT    NOT NULL UNIQUE
);

-- Many-to-many join table
CREATE TABLE IF NOT EXISTS trademark_vienna (
    trademark_id  INTEGER NOT NULL REFERENCES trademarks(id),
    vienna_code_id INTEGER NOT NULL REFERENCES vienna_codes(id),
    PRIMARY KEY (trademark_id, vienna_code_id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_tm_filename ON trademarks(filename);
CREATE INDEX IF NOT EXISTS idx_tm_year     ON trademarks(year);
CREATE INDEX IF NOT EXISTS idx_tm_split    ON trademarks(split);
CREATE INDEX IF NOT EXISTS idx_vc_code     ON vienna_codes(code);
CREATE INDEX IF NOT EXISTS idx_tv_vc       ON trademark_vienna(vienna_code_id);
CREATE INDEX IF NOT EXISTS idx_tv_tm       ON trademark_vienna(trademark_id);
"""


def build_database(entries: List[Dict], db_path: Path) -> Dict:
    """Creates normalized SQLite DB from validated entries."""
    log.info(f"Building normalized database at {db_path} ...")

    if db_path.exists():
        db_path.unlink()
        log.info("  Removed existing DB")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor = conn.cursor()

    # Create schema
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    # ── Insert trademarks ──
    log.info(f"  Inserting {len(entries)} trademarks ...")
    tm_batch = []
    for entry in entries:
        tm_batch.append((
            entry["file"],
            entry.get("text"),
            entry.get("year"),
            None,  # split assigned later
        ))

    cursor.executemany(
        "INSERT INTO trademarks (filename, tm_text, year, split) VALUES (?, ?, ?, ?)",
        tm_batch,
    )
    conn.commit()
    log.info(f"  Inserted {cursor.rowcount} trademark rows")

    # ── Build filename -> id map ──
    log.info("  Building filename->id lookup ...")
    cursor.execute("SELECT id, filename FROM trademarks")
    fname_to_id = {row[1]: row[0] for row in cursor.fetchall()}

    # ── Collect unique Vienna codes & insert ──
    log.info("  Collecting unique Vienna codes ...")
    all_codes: Set[str] = set()
    for entry in entries:
        for code in entry.get("vienna_codes", []):
            if isinstance(code, str) and code.strip():
                all_codes.add(code.strip())

    log.info(f"  Found {len(all_codes)} unique Vienna codes (full precision)")
    cursor.executemany(
        "INSERT OR IGNORE INTO vienna_codes (code) VALUES (?)",
        [(c,) for c in sorted(all_codes)],
    )
    conn.commit()

    # Build code -> id map
    cursor.execute("SELECT id, code FROM vienna_codes")
    code_to_id = {row[1]: row[0] for row in cursor.fetchall()}

    # ── Insert trademark_vienna join rows ──
    log.info("  Inserting trademark ↔ vienna_code mappings ...")
    join_batch = []
    for entry in entries:
        tm_id = fname_to_id[entry["file"]]
        for code in entry.get("vienna_codes", []):
            code = code.strip() if isinstance(code, str) else ""
            if code and code in code_to_id:
                join_batch.append((tm_id, code_to_id[code]))

        if len(join_batch) >= BATCH_SIZE:
            cursor.executemany(
                "INSERT OR IGNORE INTO trademark_vienna (trademark_id, vienna_code_id) VALUES (?, ?)",
                join_batch,
            )
            conn.commit()
            join_batch = []

    if join_batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO trademark_vienna (trademark_id, vienna_code_id) VALUES (?, ?)",
            join_batch,
        )
        conn.commit()

    total_joins = cursor.execute("SELECT COUNT(*) FROM trademark_vienna").fetchone()[0]
    log.info(f"  Inserted {total_joins} trademark↔vienna mappings")

    conn.close()

    stats = {
        "trademarks": len(entries),
        "unique_vienna_codes": len(all_codes),
        "trademark_vienna_mappings": total_joins,
        "avg_codes_per_trademark": round(total_joins / max(len(entries), 1), 2),
    }
    return stats


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Assign Train/Val/Test Splits
# ═══════════════════════════════════════════════════════════════════════

def assign_splits(db_path: Path) -> Dict:
    """Assign stratified train/val/test splits based on primary Vienna code."""
    log.info("Assigning stratified splits ...")
    random.seed(RANDOM_SEED)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get each trademark's primary vienna code (first one alphabetically)
    cursor.execute("""
        SELECT t.id, COALESCE(
            (SELECT vc.code FROM trademark_vienna tv
             JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
             WHERE tv.trademark_id = t.id
             ORDER BY vc.code LIMIT 1),
            'unknown'
        ) as primary_code
        FROM trademarks t
    """)
    rows = cursor.fetchall()
    log.info(f"  Fetched {len(rows)} trademarks for split assignment")

    # Group by level-2 code
    groups = defaultdict(list)
    for tm_id, code in rows:
        level2 = ".".join(code.split(".")[:2])
        groups[level2].append(tm_id)

    train_ids, val_ids, test_ids = [], [], []
    for code, ids in groups.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    # Batch update splits
    log.info(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for i in range(0, len(id_list), BATCH_SIZE):
            batch = id_list[i:i + BATCH_SIZE]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"UPDATE trademarks SET split = ? WHERE id IN ({placeholders})",
                [split_name] + batch,
            )
        conn.commit()

    # Verify
    cursor.execute("SELECT split, COUNT(*) FROM trademarks GROUP BY split")
    split_counts = dict(cursor.fetchall())
    log.info(f"  Split counts: {split_counts}")

    conn.close()
    return split_counts


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Compute EDA Statistics (from DB)
# ═══════════════════════════════════════════════════════════════════════

def compute_eda_stats(db_path: Path) -> Dict:
    """Compute key EDA statistics from the normalized DB."""
    log.info("Computing EDA statistics from DB ...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    stats = {}

    # Total counts
    stats["total_trademarks"] = cursor.execute("SELECT COUNT(*) FROM trademarks").fetchone()[0]
    stats["total_vienna_codes"] = cursor.execute("SELECT COUNT(*) FROM vienna_codes").fetchone()[0]
    stats["total_mappings"] = cursor.execute("SELECT COUNT(*) FROM trademark_vienna").fetchone()[0]

    # Text presence
    stats["with_text"] = cursor.execute(
        "SELECT COUNT(*) FROM trademarks WHERE tm_text IS NOT NULL AND tm_text != ''"
    ).fetchone()[0]
    stats["without_text"] = stats["total_trademarks"] - stats["with_text"]

    # Year distribution
    cursor.execute(
        "SELECT year, COUNT(*) FROM trademarks WHERE year IS NOT NULL GROUP BY year ORDER BY year"
    )
    stats["year_distribution"] = {str(row[0]): row[1] for row in cursor.fetchall()}

    # Top 20 Vienna codes at level 2
    cursor.execute("""
        SELECT SUBSTR(vc.code, 1, 5) as l2, COUNT(DISTINCT tv.trademark_id) as cnt
        FROM trademark_vienna tv
        JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
        GROUP BY l2
        ORDER BY cnt DESC
        LIMIT 20
    """)
    stats["top20_vienna_l2"] = [(row[0], row[1]) for row in cursor.fetchall()]

    # Vienna codes per trademark distribution
    cursor.execute("""
        SELECT num_codes, COUNT(*) as cnt FROM (
            SELECT trademark_id, COUNT(*) as num_codes
            FROM trademark_vienna
            GROUP BY trademark_id
        )
        GROUP BY num_codes
        ORDER BY num_codes
    """)
    stats["vienna_codes_per_tm"] = [(row[0], row[1]) for row in cursor.fetchall()]

    # Trademarks with zero vienna codes
    stats["tm_without_vienna"] = cursor.execute("""
        SELECT COUNT(*) FROM trademarks t
        WHERE NOT EXISTS (SELECT 1 FROM trademark_vienna tv WHERE tv.trademark_id = t.id)
    """).fetchone()[0]

    # Split counts
    cursor.execute("SELECT split, COUNT(*) FROM trademarks GROUP BY split")
    stats["splits"] = dict(cursor.fetchall())

    # DB file size
    stats["db_size_mb"] = round(os.path.getsize(str(db_path)) / (1024 * 1024), 1)

    conn.close()
    return stats


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Generate Markdown Report
# ═══════════════════════════════════════════════════════════════════════

def generate_report(
    quality: Dict,
    image_check: Optional[Dict],
    db_stats: Dict,
    eda_stats: Dict,
    output_path: Path,
):
    """Generate a comprehensive markdown EDA report."""
    log.info(f"Generating report at {output_path} ...")

    lines = []
    lines.append("# L3D Dataset — EDA Report v2 (Normalized Schema)")
    lines.append("")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Source**: {RAW_DIR}  ")
    lines.append(f"**Database**: `{DB_PATH}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. Data Loading Summary ──
    lines.append("## 1. Raw Data Loading")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Source JSON files** | {len(quality['per_year'])} (output_1996.json — output_2020.json) |")
    lines.append(f"| **Total raw entries** | {quality['total_raw']:,} |")
    lines.append(f"| **Unique entries loaded** | {quality['unique_entries']:,} |")
    lines.append(f"| **Duplicate filenames skipped** | {quality['duplicate_filenames']:,} |")
    lines.append(f"| **Missing `file` field** | {quality['missing_file_field']:,} |")
    lines.append(f"| **All JSON keys seen** | `{quality['all_keys_seen']}` |")
    lines.append("")

    # ── 2. Data Quality ──
    lines.append("## 2. Data Quality")
    lines.append("")
    lines.append("### Vienna Codes Format")
    lines.append(f"| Type | Count | % |")
    lines.append(f"|------|-------|---|")
    total = quality['vienna_as_string'] + quality['vienna_as_list'] + quality['vienna_none']
    for label, val in [("List (correct)", quality['vienna_as_list']),
                       ("String (normalized to list)", quality['vienna_as_string']),
                       ("None/missing", quality['vienna_none'])]:
        pct = f"{100*val/max(total,1):.1f}%"
        lines.append(f"| {label} | {val:,} | {pct} |")
    lines.append("")

    lines.append("### Text Presence")
    lines.append(f"| | Count | % |")
    lines.append(f"|--|-------|---|")
    total_t = quality['text_present'] + quality['text_null']
    lines.append(f"| With text | {quality['text_present']:,} | {100*quality['text_present']/max(total_t,1):.1f}% |")
    lines.append(f"| Without text | {quality['text_null']:,} | {100*quality['text_null']/max(total_t,1):.1f}% |")
    lines.append("")

    if quality['missing_year_field'] > 0:
        lines.append(f"⚠️ **{quality['missing_year_field']}** entries had missing/null year field")
        lines.append("")

    # ── 3. Image File Check ──
    if image_check:
        lines.append("## 3. Image File Verification")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Images on disk** | {image_check['images_on_disk']:,} |")
        lines.append(f"| **Metadata → image matched** | {image_check['metadata_matched']:,} |")
        lines.append(f"| **Metadata without image** | {image_check['metadata_missing_image']:,} |")
        lines.append(f"| **Filename case mismatches** | {image_check['case_mismatches']:,} |")
        if image_check["missing_samples"]:
            lines.append(f"| **Sample missing** | `{'`, `'.join(image_check['missing_samples'][:5])}` |")
        lines.append("")

    # ── 4. Database Schema ──
    section = 4 if image_check else 3
    lines.append(f"## {section}. Normalized Database")
    lines.append("")
    lines.append(f"| Table | Rows | Purpose |")
    lines.append(f"|-------|------|---------|")
    lines.append(f"| `trademarks` | {db_stats['trademarks']:,} | 1 row per image (filename, text, year, split) |")
    lines.append(f"| `vienna_codes` | {db_stats['unique_vienna_codes']:,} | Deduplicated Vienna code lookup |")
    lines.append(f"| `trademark_vienna` | {db_stats['trademark_vienna_mappings']:,} | Many-to-many join (avg {db_stats['avg_codes_per_trademark']} codes/trademark) |")
    lines.append(f"| **DB file size** | {eda_stats['db_size_mb']} MB | vs ~488 MB before (3 files combined) |")
    lines.append("")

    lines.append("```sql")
    lines.append(SCHEMA_SQL.strip())
    lines.append("```")
    lines.append("")

    # ── 5. Year Distribution ──
    section += 1
    lines.append(f"## {section}. Year Distribution")
    lines.append("")
    lines.append("| Year | Count | Year | Count | Year | Count |")
    lines.append("|------|-------|------|-------|------|-------|")
    years = list(eda_stats["year_distribution"].items())
    # Arrange in 3 columns
    n_rows = (len(years) + 2) // 3
    for r in range(n_rows):
        cols = []
        for c in range(3):
            idx = r + c * n_rows
            if idx < len(years):
                yr, cnt = years[idx]
                cols.append(f"| {yr} | {cnt:,} ")
            else:
                cols.append("| | ")
        lines.append("".join(cols) + "|")
    lines.append("")

    # ── 6. Top Vienna Codes ──
    section += 1
    lines.append(f"## {section}. Top 20 Vienna Codes (Level 2)")
    lines.append("")
    lines.append("| Rank | Code | Trademarks | % of Dataset |")
    lines.append("|------|------|------------|--------------|")
    total_tm = eda_stats["total_trademarks"]
    for i, (code, cnt) in enumerate(eda_stats["top20_vienna_l2"], 1):
        pct = f"{100*cnt/max(total_tm,1):.1f}%"
        lines.append(f"| {i} | {code} | {cnt:,} | {pct} |")
    lines.append("")

    # ── 7. Vienna codes per trademark ──
    section += 1
    lines.append(f"## {section}. Vienna Codes per Trademark")
    lines.append("")
    lines.append("| # Codes | # Trademarks |")
    lines.append("|---------|-------------|")
    for n_codes, cnt in eda_stats["vienna_codes_per_tm"][:15]:
        lines.append(f"| {n_codes} | {cnt:,} |")
    if eda_stats["tm_without_vienna"] > 0:
        lines.append(f"| 0 (no codes) | {eda_stats['tm_without_vienna']:,} |")
    lines.append("")

    # ── 8. Splits ──
    section += 1
    lines.append(f"## {section}. Data Splits")
    lines.append("")
    lines.append("| Split | Count | % |")
    lines.append("|-------|-------|---|")
    for split_name in ["train", "val", "test"]:
        cnt = eda_stats["splits"].get(split_name, 0)
        pct = f"{100*cnt/max(total_tm,1):.1f}%"
        lines.append(f"| {split_name} | {cnt:,} | {pct} |")
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Report written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Export Splits as JSON (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

def export_splits_json(db_path: Path, output_dir: Path):
    """Export train/val/test splits as JSON for backward compat with TrademarkDataset."""
    log.info("Exporting splits as JSON for backward compatibility ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for split_name in ["train", "val", "test"]:
        cursor.execute("""
            SELECT t.filename, t.tm_text, t.year,
                   GROUP_CONCAT(vc.code, '|||') as codes
            FROM trademarks t
            LEFT JOIN trademark_vienna tv ON tv.trademark_id = t.id
            LEFT JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
            WHERE t.split = ?
            GROUP BY t.id
        """, (split_name,))

        entries = []
        for row in cursor.fetchall():
            codes_str = row[3]
            codes = codes_str.split("|||") if codes_str else []
            entries.append({
                "file": row[0],
                "text": row[1],
                "vienna_codes": codes,
                "year": row[2],
            })

        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)
        log.info(f"  {split_name}: {len(entries):,} entries -> {out_path}")

    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EDA + Schema Migration")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip checking image file existence (faster)")
    parser.add_argument("--skip-json-export", action="store_true",
                        help="Skip exporting splits as JSON files")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("STARTING EDA + SCHEMA MIGRATION")
    log.info(f"RAW_DATASET_DIR = {RAW_DIR}")
    log.info(f"DB_PATH         = {DB_PATH}")
    log.info("=" * 60)

    # Validate paths
    if not RAW_DIR.exists():
        log.error(f"RAW_DATASET_DIR does not exist: {RAW_DIR}")
        sys.exit(1)

    images_dir = RAW_DIR / "images"
    if not images_dir.exists():
        log.error(f"Images directory does not exist: {images_dir}")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load & validate
    entries, quality = load_yearly_jsons(RAW_DIR)

    # Step 2: Image check
    image_check = None
    if not args.skip_image_check:
        image_check = check_image_existence(entries, images_dir)

    # Step 3: Build DB
    db_stats = build_database(entries, DB_PATH)

    # Step 4: Assign splits
    split_counts = assign_splits(DB_PATH)

    # Step 5: Compute EDA stats
    eda_stats = compute_eda_stats(DB_PATH)

    # Step 6: Generate report
    report_path = DOCS_DIR / "eda_report_v2.md"
    generate_report(quality, image_check, db_stats, eda_stats, report_path)

    # Step 7: Export JSON splits (backward compat)
    if not args.skip_json_export:
        export_splits_json(DB_PATH, DATA_DIR / "splits")

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"DONE in {elapsed:.1f}s")
    log.info(f"Database: {DB_PATH} ({eda_stats['db_size_mb']} MB)")
    log.info(f"Report:   {report_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
