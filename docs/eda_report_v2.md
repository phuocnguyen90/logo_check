# L3D Dataset — EDA Report v2 (Normalized Schema)

**Generated**: 2026-02-27 14:22  
**Source**: /home/phuoc/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset  
**Database**: `/home/phuoc/git/l3d/tm-dataset/data/metadata_v2.db`  

---

## 1. Raw Data Loading

| Metric | Value |
|--------|-------|
| **Source JSON files** | 25 (output_1996.json — output_2020.json) |
| **Total raw entries** | 769,773 |
| **Unique entries loaded** | 769,773 |
| **Duplicate filenames skipped** | 0 |
| **Missing `file` field** | 0 |
| **All JSON keys seen** | `['file', 'text', 'vienna_codes', 'year']` |

## 2. Data Quality

### Vienna Codes Format
| Type | Count | % |
|------|-------|---|
| List (correct) | 572,013 | 74.3% |
| String (normalized to list) | 197,760 | 25.7% |
| None/missing | 0 | 0.0% |

### Text Presence
| | Count | % |
|--|-------|---|
| With text | 693,355 | 90.1% |
| Without text | 76,418 | 9.9% |

## 3. Image File Verification

| Metric | Value |
|--------|-------|
| **Images on disk** | 769,674 |
| **Metadata → image matched** | 769,674 |
| **Metadata without image** | 99 |
| **Filename case mismatches** | 769,674 |
| **Sample missing** | `0c70135e-9902-483c-924c-3485bdc45598.JPG`, `e79c578d-cfae-433f-bb4d-567ecdb1ede0.JPG`, `e9dcb1cc-0eeb-451e-ae39-88a1f2d14db3.JPG`, `75bb21a6-5aa7-4ac1-bd6d-4ec4769067be.TIF`, `3da9b8a7-6475-4924-a114-e474fd22fcb8.mp3` |

## 4. Normalized Database

| Table | Rows | Purpose |
|-------|------|---------|
| `trademarks` | 769,773 | 1 row per image (filename, text, year, split) |
| `vienna_codes` | 1,558 | Deduplicated Vienna code lookup |
| `trademark_vienna` | 2,454,417 | Many-to-many join (avg 3.19 codes/trademark) |
| **DB file size** | 298.8 MB | vs ~488 MB before (3 files combined) |

```sql
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
```

## 5. Year Distribution

| Year | Count | Year | Count | Year | Count |
|------|-------|------|-------|------|-------|
| 1996 | 15,138 | 2005 | 21,740 | 2014 | 43,488 |
| 1997 | 9,518 | 2006 | 24,981 | 2015 | 48,747 |
| 1998 | 11,407 | 2007 | 28,916 | 2016 | 54,603 |
| 1999 | 14,474 | 2008 | 28,765 | 2017 | 54,576 |
| 2000 | 19,188 | 2009 | 29,699 | 2018 | 57,575 |
| 2001 | 16,172 | 2010 | 33,721 | 2019 | 59,824 |
| 2002 | 15,613 | 2011 | 35,815 | 2020 | 53,123 |
| 2003 | 20,852 | 2012 | 10,363 | | |
| 2004 | 21,569 | 2013 | 39,906 | | |

## 6. Top 20 Vienna Codes (Level 2)

| Rank | Code | Trademarks | % of Dataset |
|------|------|------------|--------------|
| 1 | 27.05 | 221,101 | 28.7% |
| 2 | 29.01 | 138,336 | 18.0% |
| 3 | 26.04 | 97,837 | 12.7% |
| 4 | 27.99 | 94,188 | 12.2% |
| 5 | 26.11 | 77,883 | 10.1% |
| 6 | 25.05 | 70,448 | 9.2% |
| 7 | 26.01 | 65,882 | 8.6% |
| 8 | 27.03 | 36,323 | 4.7% |
| 9 | 24.17 | 35,471 | 4.6% |
| 10 | 01.15 | 34,989 | 4.5% |
| 11 | 02.01 | 29,105 | 3.8% |
| 12 | 25.01 | 28,295 | 3.7% |
| 13 | 26.07 | 26,062 | 3.4% |
| 14 | 02.09 | 24,506 | 3.2% |
| 15 | 26.03 | 23,796 | 3.1% |
| 16 | 25.07 | 22,788 | 3.0% |
| 17 | 27.01 | 22,584 | 2.9% |
| 18 | 05.03 | 21,809 | 2.8% |
| 19 | 26.99 | 21,157 | 2.7% |
| 20 | 01.01 | 18,213 | 2.4% |

## 7. Vienna Codes per Trademark

| # Codes | # Trademarks |
|---------|-------------|
| 1 | 197,760 |
| 2 | 160,901 |
| 3 | 144,234 |
| 4 | 93,812 |
| 5 | 64,502 |
| 6 | 43,500 |
| 7 | 27,426 |
| 8 | 16,133 |
| 9 | 9,125 |
| 10 | 5,383 |
| 11 | 3,105 |
| 12 | 1,632 |
| 13 | 1,002 |
| 14 | 521 |
| 15 | 284 |

## 8. Data Splits

| Split | Count | % |
|-------|-------|---|
| train | 538,771 | 70.0% |
| val | 115,392 | 15.0% |
| test | 115,610 | 15.0% |
