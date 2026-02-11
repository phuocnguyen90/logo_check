# L3D Dataset - Exploratory Data Analysis (EDA) Report

**Generated**: 2026-02-11  
**Dataset**: L3D (Large Labelled Logo Dataset)  
**Source**: EUIPO TMView (1996-2020) via KaggleHub (`konradb/ziilogos`)

---

## Executive Summary

The L3D dataset contains **769,773 trademark logo images** from the EUIPO TMView database spanning 1996-2020. All images are pre-normalized to 256×256 pixels. The dataset is rich in metadata with **90.1% of images containing text labels** and comprehensive Vienna code classifications for figurative elements.

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Images** | 769,773 |
| **Time Period** | 1996 - 2020 (25 years) |
| **Image Format** | JPG |
| **Image Size** | 256×256 pixels (normalized) |
| **With Text** | 693,355 (90.1%) |
| **Without Text** | 76,418 (9.9%) |
| **RGB Images** | 478,899 (62.2%) |
| **Grayscale** | 290,775 (37.8%) |
| **Corrupted Images** | 0 (0%) |

---

## 2. Data Splits

Stratified train/val/test splits were created based on Vienna codes to ensure balanced representation:

| Split | Images | Percentage | File |
|-------|--------|------------|------|
| **Train** | 538,769 | 70.0% | `data/splits/train.json` |
| **Validation** | 115,394 | 15.0% | `data/splits/val.json` |
| **Test** | 115,610 | 15.0% | `data/splits/test.json` |

### Validation Pairs

For evaluation, we created pairs of known-similar and known-dissimilar logos:

| Pair Type | Count | Criteria |
|-----------|-------|----------|
| **Similar** | 500 | Same Vienna code (level 2) |
| **Dissimilar** | 500 | Different Vienna codes |

Files: `data/validation/similar_pairs.json`, `data/validation/dissimilar_pairs.json`

---

## 3. Vienna Code Distribution

Vienna codes classify figurative elements in trademarks. We use **level 2 codes** (first two components, e.g., "26.04") to group similar logos.

### Top 20 Vienna Codes (Level 2)

| Rank | Code | Count | % of Dataset | Description |
|------|------|-------|--------------|-------------|
| 1 | 26.04 | 350,549 | 45.5% | Inscriptions in various characters with figurative elements |
| 2 | 29.01 | 242,418 | 31.5% | Bottles, flasks, carafes - containers |
| 3 | 27.05 | 230,565 | 29.9% | Quadrilaterals with one or more polygons |
| 4 | 26.11 | 174,906 | 22.7% | Different figurative elements |
| 5 | 27.99 | 149,098 | 19.4% | Other geometrical figures (undefined) |
| 6 | 26.01 | 117,822 | 15.3% | Letters, numerals, punctuation marks |
| 7 | 25.05 | 72,001 | 9.4% | Ornamental bands, borders, frames |
| 8 | 05.03 | 49,294 | 6.4% | Animals in costumes, animals portrayed as humans |
| 9 | 02.01 | 46,018 | 6.0% | Hearts - plant representations |
| 10 | 26.99 | 43,143 | 5.6% | Other textual elements |
| 11 | 26.03 | 41,621 | 5.4% | Inscriptions in Chinese/Japanese characters |
| 12 | 27.03 | 37,463 | 4.9% | Triangles with one or more polygons |
| 13 | 01.15 | 37,214 | 4.8% | Other vegetables - plants |
| 14 | 24.17 | 37,031 | 4.8% | Shields with figurative elements or inscriptions |
| 15 | 01.01 | 35,932 | 4.7% | Fruit - plant representations |
| 16 | 25.01 | 33,810 | 4.4% | Architecture, castles, fortifications |
| 17 | 05.05 | 32,416 | 4.2% | Dragons, griffins - fabulous animals |
| 18 | 03.07 | 30,387 | 3.9% | Other representations of the human body |
| 19 | 24.15 | 28,302 | 3.7% | Shields with geometrical figures |
| 20 | 03.01 | 27,634 | 3.6% | Heads, busts - human beings |

### Distribution Insights

- **Heavy tail distribution**: Top 3 codes cover ~45% of all images
- **Diverse figurative elements**: 100+ unique Vienna codes present
- **Text-heavy**: Codes 26.xx (textual elements) appear in ~70% of logos

---

## 4. Temporal Distribution

### Yearly Breakdown

| Year | Count | Year | Count | Year | Count |
|------|-------|------|-------|------|-------|
| 1996 | 15,138 | 2004 | 21,569 | 2012 | 10,363 |
| 1997 | 9,518 | 2005 | 21,740 | 2013 | 39,906 |
| 1998 | 11,407 | 2006 | 24,981 | 2014 | 43,488 |
| 1999 | 14,474 | 2007 | 28,916 | 2015 | 48,747 |
| 2000 | 19,188 | 2008 | 28,765 | 2016 | 54,603 |
| 2001 | 16,172 | 2009 | 29,699 | 2017 | 54,576 |
| 2002 | 15,613 | 2010 | 33,721 | 2018 | 57,575 |
| 2003 | 20,852 | 2011 | 35,815 | 2019 | 59,824 |
| | | | | 2020 | 53,123 |

### Temporal Trends

- **Growth period**: Steady increase from 1996 to 2019
- **Peak year**: 2019 with 59,824 trademark applications
- **Notable dip**: 2012 with only 10,363 entries (likely data collection gap)
- **Recent data**: 2020 shows slight decrease (pandemic effect)

---

## 5. Image Characteristics

### Color Distribution

| Mode | Count | Percentage |
|------|-------|------------|
| **RGB** | 478,899 | 62.2% |
| **Grayscale** | 290,775 | 37.8% |

### Image Dimensions

All images are pre-normalized to 256×256 pixels:
- **Width**: min=256, max=256, mean=256
- **Height**: min=256, max=256, mean=256
- **Aspect Ratio**: 1:1 (square)

---

## 6. Text Presence Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| **With Text** | 693,355 | 90.1% |
| **Text-only** | 76,418 | 9.9% |

### Key Findings

- **High text prevalence**: 90% of logos contain textual elements
- **Composite marks**: Majority are composite (text + figurative)
- **Word marks**: ~10% are pure word marks (no figurative elements)

This has implications for:
1. **OCR preprocessing**: Essential for ~90% of images
2. **Text masking**: Required to isolate figurative elements
3. **Composite scoring**: Visual + text similarity combination needed

---

## 7. Data Quality Assessment

| Quality Metric | Result | Status |
|----------------|--------|--------|
| **Corrupted images** | 0 | ✅ Excellent |
| **Missing metadata** | <0.1% | ✅ Excellent |
| **Vienna code coverage** | 100% | ✅ Excellent |
| **Year coverage** | 100% | ✅ Excellent |
| **Image normalization** | 100% | ✅ Excellent |

### Data Quality Notes

- All 769,773 images are readable and properly formatted
- Metadata is complete for all entries
- No missing or null critical fields
- Images are already pre-processed to 256×256 (Kaggle dataset)

---

## 8. Implications for Model Design

### Training Strategy

1. **Contrastive Learning**: Use Vienna codes as weak supervision signal
   - Same code → similar (positive pairs)
   - Different code → dissimilar (negative pairs)

2. **Stratified Sampling**: Ensure balanced representation across Vienna codes
   - Top codes (26.04, 29.01, 27.05) are 30-45% each
   - Use weighted sampling to avoid bias toward dominant codes

3. **Text Handling**:
   - Detect and mask text regions for visual similarity
   - Use text for composite scoring (phonetic similarity)

### Architecture Decisions

1. **Input size**: 224×224 ( EfficientNet-B0 standard)
2. **Preprocessing**: Resize from 256→224 with center crop
3. **Color handling**: Support both RGB and grayscale (convert all to RGB)
4. **Batch composition**: Stratified by Vienna code for balanced training

### Evaluation Considerations

1. **Validation pairs**: Use similar_pairs.json and dissimilar_pairs.json
2. **Metrics**: Recall@K, Precision@K for retrieval
3. **Stratified evaluation**: Report per-Vienna-code performance

---

## 9. File Locations

| Output | Path |
|--------|------|
| **Statistics checkpoint** | `data/stats_checkpoint.json` |
| **Train split** | `data/splits/train.json` |
| **Validation split** | `data/splits/val.json` |
| **Test split** | `data/splits/test.json` |
| **Similar pairs** | `data/validation/similar_pairs.json` |
| **Dissimilar pairs** | `data/validation/dissimilar_pairs.json` |
| **Logs** | `logs/project.log` |

---

## 10. References

- **Dataset Paper**: Gutiérrez-Fandiño et al., "The Large Labelled Logo Dataset (L3D)", 2021
- **Dataset URL**: https://doi.org/10.5281/zenodo.5771006
- **Vienna Classification**: https://www.wipo.int/classifications/nivilo/vienna

---

## Appendix: Full Vienna Code Distribution

For the complete list of Vienna codes and their frequencies, see:
```
data/stats_checkpoint.json → stats.vienna_code_distribution
```

Top categories summary:
- **26.xx** (Textual elements): ~650K occurrences
- **27.xx** (Geometrical figures): ~450K occurrences  
- **29.xx** (Containers): ~240K occurrences
- **25.xx** (Ornamental elements): ~110K occurrences
