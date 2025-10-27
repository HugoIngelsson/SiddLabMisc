# Google Street View Tree Species Classification Evaluation Report

Generated: 2025-07-22 11:31:59

## Dataset Summary
- Total test samples: 855
- Total species classes: 210
- Classes present in test set: 26

## Overall Performance Metrics
- **Accuracy**: 0.5789
- **Balanced Accuracy**: 0.2407
- **F1 Score (Macro)**: 0.2484
- **F1 Score (Weighted)**: 0.5503
- **Top-5 Accuracy**: 0.0000
- **Mean Confidence**: 0.7322 ± 0.2276

### Precision and Recall
- Precision (Macro): 0.2834
- Precision (Weighted): 0.5335
- Recall (Macro): 0.2407
- Recall (Weighted): 0.5789

## Top Performing Classes

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Vachellia nilotica | 0.679 | 0.619 | 0.753 | 190 |
| Azadirachta indica | 0.644 | 0.608 | 0.685 | 251 |
| Ailanthus excelsa | 0.644 | 0.631 | 0.657 | 99 |
| prosopis juliflora | 0.642 | 0.619 | 0.667 | 39 |
| saraca asoca | 0.582 | 0.615 | 0.552 | 29 |
| Morus Alba | 0.545 | 0.750 | 0.429 | 21 |
| Prosopis cineraria | 0.531 | 0.495 | 0.571 | 91 |
| Aegle Marmelos | 0.421 | 0.500 | 0.364 | 11 |
| Albizia lebbeck | 0.333 | 1.000 | 0.200 | 5 |

## Lowest Performing Classes (with samples)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Ficus benghalensis | 0.000 | 0.000 | 0.000 | 12 |
| Ficus religiosa | 0.000 | 0.000 | 0.000 | 11 |
| Mangifera indica | 0.000 | 0.000 | 0.000 | 10 |
| Moringa oleifera | 0.000 | 0.000 | 0.000 | 8 |
| Phoenix sylvestris | 0.000 | 0.000 | 0.000 | 4 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 8 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 3 |

## Error Analysis
- Total errors: 360
- Error rate: 0.4211
- Mean confidence on errors: 0.6177
- Mean confidence on correct: 0.8154

### Most Confused Class Pairs

| True Class | Predicted Class | Count |
|------------|-----------------|-------|
| Azadirachta indica | Vachellia nilotica | 28 |
| Vachellia nilotica | Azadirachta indica | 20 |
| Prosopis cineraria | Vachellia nilotica | 18 |
| Azadirachta indica | Prosopis cineraria | 18 |
| Ailanthus excelsa | Azadirachta indica | 15 |
| Dalbergia sissoo | Azadirachta indica | 14 |
| Azadirachta indica | Ailanthus excelsa | 14 |
| Vachellia nilotica | Prosopis cineraria | 14 |
| Ailanthus excelsa | Vachellia nilotica | 11 |
| Prosopis cineraria | Azadirachta indica | 11 |