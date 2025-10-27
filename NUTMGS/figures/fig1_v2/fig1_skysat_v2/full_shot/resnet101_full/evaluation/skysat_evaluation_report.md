# SkySat Tree Species Classification Evaluation Report

Generated: 2025-07-30 23:20:39

## Dataset Summary
- Total SkySat test samples: 1465
- Total species classes: 223
- Classes present in test set: 28
- Modality: SkySat satellite imagery (3m resolution)

## Overall Performance Metrics
- **Accuracy**: 0.3563
- **Balanced Accuracy**: 0.0422
- **F1 Score (Macro)**: 0.0304
- **F1 Score (Weighted)**: 0.2463
- **Top-5 Accuracy**: 0.0000
- **Mean Confidence**: 0.3464 ± 0.0509

### Precision and Recall
- Precision (Macro): 0.0335
- Precision (Weighted): 0.2216
- Recall (Macro): 0.0422
- Recall (Weighted): 0.3563

## Top Performing Classes (SkySat)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Azadirachta indica | 0.511 | 0.368 | 0.838 | 488 |
| Vachellia nilotica | 0.328 | 0.321 | 0.336 | 333 |
| Prosopis cineraria | 0.013 | 0.250 | 0.007 | 153 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 5 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Albizia procera | 0.000 | 0.000 | 0.000 | 9 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Butea monosperma | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 2 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |

## Lowest Performing Classes (SkySat, with samples)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 9 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 12 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |
| Aegle Marmelos | 0.000 | 0.000 | 0.000 | 17 |
| Morus Alba | 0.000 | 0.000 | 0.000 | 34 |
| Ailanthus excelsa | 0.000 | 0.000 | 0.000 | 140 |
| saraca asoca | 0.000 | 0.000 | 0.000 | 38 |
| prosopis juliflora | 0.000 | 0.000 | 0.000 | 65 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 9 |

## Error Analysis
- Total errors: 943
- Error rate: 0.6437
- Mean confidence on errors: 0.3425
- Mean confidence on correct: 0.3535

### Most Confused Class Pairs (SkySat)

| True Class | Predicted Class | Count |
|------------|-----------------|-------|
| Vachellia nilotica | Azadirachta indica | 220 |
| Ailanthus excelsa | Azadirachta indica | 106 |
| Prosopis cineraria | Azadirachta indica | 100 |
| Azadirachta indica | Vachellia nilotica | 79 |
| prosopis juliflora | Azadirachta indica | 54 |
| Prosopis cineraria | Vachellia nilotica | 52 |
| Ailanthus excelsa | Vachellia nilotica | 34 |
| saraca asoca | Azadirachta indica | 33 |
| Dalbergia sissoo | Azadirachta indica | 30 |
| Morus Alba | Azadirachta indica | 29 |

## SkySat-Specific Notes
- **Spatial Resolution**: 3-meter pixels provide fine-scale tree canopy details
- **Spectral Bands**: RGB imagery optimized for visual interpretation
- **Temporal Consistency**: Single acquisition reduces seasonal variation effects
- **Challenges**: Limited spectral information compared to multispectral sensors