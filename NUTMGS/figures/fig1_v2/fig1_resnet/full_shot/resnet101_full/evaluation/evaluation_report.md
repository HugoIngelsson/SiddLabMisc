# Google Street View Tree Species Classification Evaluation Report

Generated: 2025-07-31 00:48:53

## Dataset Summary
- Total test samples: 1465
- Total species classes: 223
- Classes present in test set: 28

## Overall Performance Metrics
- **Accuracy**: 0.6485
- **Balanced Accuracy**: 0.2969
- **F1 Score (Macro)**: 0.3061
- **F1 Score (Weighted)**: 0.6239
- **Top-5 Accuracy**: 0.0000
- **Mean Confidence**: 0.7448 ± 0.2240

### Precision and Recall
- Precision (Macro): 0.3406
- Precision (Weighted): 0.6138
- Recall (Macro): 0.2969
- Recall (Weighted): 0.6485

## Top Performing Classes

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Butea monosperma | 1.000 | 1.000 | 1.000 | 1 |
| prosopis juliflora | 0.813 | 0.862 | 0.769 | 65 |
| Azadirachta indica | 0.753 | 0.689 | 0.830 | 488 |
| Phoenix sylvestris | 0.727 | 0.615 | 0.889 | 9 |
| Vachellia nilotica | 0.727 | 0.684 | 0.775 | 333 |
| saraca asoca | 0.650 | 0.619 | 0.684 | 38 |
| Ailanthus excelsa | 0.581 | 0.616 | 0.550 | 140 |
| Prosopis cineraria | 0.542 | 0.563 | 0.523 | 153 |
| Pongamia pinnata | 0.516 | 0.500 | 0.533 | 15 |
| Eucalyptus tereticornis | 0.462 | 0.750 | 0.333 | 9 |

## Lowest Performing Classes (with samples)

| Class | F1 Score | Precision | Recall | Support |
|-------|----------|-----------|---------|---------|
| Aegle Marmelos | 0.083 | 0.143 | 0.059 | 17 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 5 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 2 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 9 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis
- Total errors: 515
- Error rate: 0.3515
- Mean confidence on errors: 0.6232
- Mean confidence on correct: 0.8108

### Most Confused Class Pairs

| True Class | Predicted Class | Count |
|------------|-----------------|-------|
| Vachellia nilotica | Azadirachta indica | 36 |
| Prosopis cineraria | Vachellia nilotica | 36 |
| Azadirachta indica | Vachellia nilotica | 32 |
| Ailanthus excelsa | Azadirachta indica | 30 |
| Prosopis cineraria | Azadirachta indica | 28 |
| Dalbergia sissoo | Azadirachta indica | 23 |
| Vachellia nilotica | Prosopis cineraria | 18 |
| Azadirachta indica | Ailanthus excelsa | 17 |
| Azadirachta indica | Prosopis cineraria | 15 |
| Ailanthus excelsa | Vachellia nilotica | 15 |