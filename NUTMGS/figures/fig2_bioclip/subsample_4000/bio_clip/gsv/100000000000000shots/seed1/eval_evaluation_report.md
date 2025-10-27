# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 15:22:06

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6224
- **Balanced Accuracy**: 0.3433
- **F1 Score (Macro)**: 0.3517
- **F1 Score (Weighted)**: 0.6073
- **Precision (Macro)**: 0.4025
- **Recall (Macro)**: 0.3306
- **Top-1 Accuracy**: 0.6216
- **Top-3 Accuracy**: 0.8716
- **Top-5 Accuracy**: 0.9241

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0053, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Azadirachta indica | 0.726 | 0.649 | 0.823 | 344 |
| Vachellia nilotica | 0.692 | 0.680 | 0.704 | 257 |
| Phoenix sylvestris | 0.667 | 0.750 | 0.600 | 5 |
| Ailanthus excelsa | 0.656 | 0.697 | 0.619 | 134 |
| prosopis juliflora | 0.623 | 0.589 | 0.660 | 50 |
| saraca asoca | 0.576 | 0.739 | 0.472 | 36 |
| Eucalyptus tereticornis | 0.571 | 1.000 | 0.400 | 5 |
| Pongamia pinnata | 0.533 | 0.500 | 0.571 | 14 |
| Morus Alba | 0.510 | 0.619 | 0.433 | 30 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Ziziphus jujuba | 0.267 | 0.667 | 0.167 | 12 |
| Moringa oleifera | 0.125 | 0.143 | 0.111 | 9 |
| Mangifera indica | 0.111 | 0.200 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 8 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 438
- Error rate: 0.3776
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 29 |
| Prosopis cineraria | Azadirachta indica | 28 |
| Ailanthus excelsa | Azadirachta indica | 27 |
| Vachellia nilotica | Azadirachta indica | 27 |
| Vachellia nilotica | Prosopis cineraria | 22 |
| Azadirachta indica | Vachellia nilotica | 17 |
| Ailanthus excelsa | Vachellia nilotica | 15 |
| Vachellia nilotica | prosopis juliflora | 14 |
| Azadirachta indica | Prosopis cineraria | 13 |
| Dalbergia sissoo | Azadirachta indica | 12 |