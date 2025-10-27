# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 20:01:43

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6328
- **Balanced Accuracy**: 0.3794
- **F1 Score (Macro)**: 0.3774
- **F1 Score (Weighted)**: 0.6218
- **Precision (Macro)**: 0.4327
- **Recall (Macro)**: 0.3653
- **Top-1 Accuracy**: 0.6371
- **Top-3 Accuracy**: 0.8647
- **Top-5 Accuracy**: 0.9250

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Eucalyptus tereticornis | 0.750 | 1.000 | 0.600 | 5 |
| Azadirachta indica | 0.733 | 0.670 | 0.808 | 344 |
| Vachellia nilotica | 0.688 | 0.690 | 0.685 | 257 |
| prosopis juliflora | 0.673 | 0.632 | 0.720 | 50 |
| Ailanthus excelsa | 0.667 | 0.669 | 0.664 | 134 |
| Cassia fistula | 0.667 | 0.500 | 1.000 | 1 |
| saraca asoca | 0.606 | 0.667 | 0.556 | 36 |
| Ficus benghalensis | 0.571 | 0.615 | 0.533 | 15 |
| Aegle Marmelos | 0.552 | 0.571 | 0.533 | 15 |
| Prosopis cineraria | 0.512 | 0.542 | 0.485 | 134 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia lebbeck | 0.286 | 0.500 | 0.200 | 5 |
| Moringa oleifera | 0.286 | 0.400 | 0.222 | 9 |
| Ziziphus jujuba | 0.267 | 0.667 | 0.167 | 12 |
| Vachellia leucophloea | 0.167 | 0.250 | 0.125 | 8 |
| Mangifera indica | 0.091 | 0.111 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 426
- Error rate: 0.3672
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 30 |
| Ailanthus excelsa | Azadirachta indica | 26 |
| Vachellia nilotica | Azadirachta indica | 26 |
| Vachellia nilotica | Prosopis cineraria | 25 |
| Prosopis cineraria | Azadirachta indica | 20 |
| Azadirachta indica | Prosopis cineraria | 16 |
| Azadirachta indica | Ailanthus excelsa | 15 |
| Vachellia nilotica | prosopis juliflora | 15 |
| Azadirachta indica | Vachellia nilotica | 11 |
| Ailanthus excelsa | Vachellia nilotica | 11 |