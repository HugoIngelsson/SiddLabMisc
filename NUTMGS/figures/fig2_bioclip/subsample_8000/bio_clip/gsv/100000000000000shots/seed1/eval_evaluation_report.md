# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-25 02:12:25

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6560
- **Balanced Accuracy**: 0.3974
- **F1 Score (Macro)**: 0.4203
- **F1 Score (Weighted)**: 0.6450
- **Precision (Macro)**: 0.4676
- **Recall (Macro)**: 0.3974
- **Top-1 Accuracy**: 0.6552
- **Top-3 Accuracy**: 0.8793
- **Top-5 Accuracy**: 0.9379

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Phoenix sylvestris | 0.889 | 1.000 | 0.800 | 5 |
| Azadirachta indica | 0.750 | 0.691 | 0.820 | 344 |
| Vachellia nilotica | 0.713 | 0.685 | 0.743 | 257 |
| Ailanthus excelsa | 0.682 | 0.701 | 0.664 | 134 |
| Eucalyptus tereticornis | 0.667 | 0.750 | 0.600 | 5 |
| prosopis juliflora | 0.653 | 0.647 | 0.660 | 50 |
| saraca asoca | 0.635 | 0.741 | 0.556 | 36 |
| Morus Alba | 0.600 | 0.750 | 0.500 | 30 |
| Prosopis cineraria | 0.534 | 0.584 | 0.493 | 134 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Vachellia leucophloea | 0.333 | 0.500 | 0.250 | 8 |
| Albizia procera | 0.300 | 0.250 | 0.375 | 8 |
| Moringa oleifera | 0.133 | 0.167 | 0.111 | 9 |
| Mangifera indica | 0.100 | 0.143 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 399
- Error rate: 0.3440
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0058

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 30 |
| Prosopis cineraria | Azadirachta indica | 23 |
| Vachellia nilotica | Prosopis cineraria | 21 |
| Vachellia nilotica | Azadirachta indica | 21 |
| Ailanthus excelsa | Azadirachta indica | 20 |
| Azadirachta indica | Vachellia nilotica | 18 |
| Ailanthus excelsa | Vachellia nilotica | 15 |
| Azadirachta indica | Ailanthus excelsa | 13 |
| Azadirachta indica | Prosopis cineraria | 12 |
| saraca asoca | Azadirachta indica | 12 |