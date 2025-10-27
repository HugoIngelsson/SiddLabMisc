# SkySat Tree Species Classification Evaluation Report

Generated: 2025-07-22 13:24:03

## Dataset Summary
- Total SkySat test samples: 855
- Total species classes: 210
- Classes present in test set: 26
- Modality: SkySat satellite imagery (3m resolution)

## Overall Performance Metrics
- **Accuracy**: 0.3135
- **Balanced Accuracy**: 0.0431
- **F1 Score (Macro)**: 0.0280
- **F1 Score (Weighted)**: 0.1947
- **Top-5 Accuracy**: 0.0000
- **Mean Confidence**: 0.2955 ± 0.0435

### Precision and Recall
- Precision (Macro): 0.0251
- Precision (Weighted): 0.1670
- Recall (Macro): 0.0431
- Recall (Weighted): 0.3135

## Top Performing Classes (SkySat)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Azadirachta indica | 0.459 | 0.308 | 0.900 | 251 |
| Vachellia nilotica | 0.269 | 0.344 | 0.221 | 190 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 3 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Albizia procera | 0.000 | 0.000 | 0.000 | 8 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Dalbergia sissoo | 0.000 | 0.000 | 0.000 | 27 |
| Ficus benghalensis | 0.000 | 0.000 | 0.000 | 12 |

## Lowest Performing Classes (SkySat, with samples)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 8 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |
| Aegle Marmelos | 0.000 | 0.000 | 0.000 | 11 |
| Morus Alba | 0.000 | 0.000 | 0.000 | 21 |
| Ailanthus excelsa | 0.000 | 0.000 | 0.000 | 99 |
| saraca asoca | 0.000 | 0.000 | 0.000 | 29 |
| prosopis juliflora | 0.000 | 0.000 | 0.000 | 39 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 3 |

## Error Analysis
- Total errors: 587
- Error rate: 0.6865
- Mean confidence on errors: 0.2922
- Mean confidence on correct: 0.3028

### Most Confused Class Pairs (SkySat)

| True Class | Predicted Class | Count |
|------------|-----------------|-------|
| Vachellia nilotica | Azadirachta indica | 148 |
| Ailanthus excelsa | Azadirachta indica | 83 |
| Prosopis cineraria | Azadirachta indica | 79 |
| prosopis juliflora | Azadirachta indica | 33 |
| saraca asoca | Azadirachta indica | 29 |
| Azadirachta indica | Vachellia nilotica | 25 |
| Dalbergia sissoo | Azadirachta indica | 20 |
| Morus Alba | Azadirachta indica | 20 |
| Ailanthus excelsa | Vachellia nilotica | 16 |
| Prosopis cineraria | Vachellia nilotica | 12 |

## SkySat-Specific Notes
- **Spatial Resolution**: 3-meter pixels provide fine-scale tree canopy details
- **Spectral Bands**: RGB imagery optimized for visual interpretation
- **Temporal Consistency**: Single acquisition reduces seasonal variation effects
- **Challenges**: Limited spectral information compared to multispectral sensors