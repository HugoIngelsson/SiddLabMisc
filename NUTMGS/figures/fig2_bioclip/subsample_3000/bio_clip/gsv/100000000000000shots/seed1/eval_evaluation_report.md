# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 13:34:50

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6190
- **Balanced Accuracy**: 0.3710
- **F1 Score (Macro)**: 0.3996
- **F1 Score (Weighted)**: 0.6041
- **Precision (Macro)**: 0.4726
- **Recall (Macro)**: 0.3710
- **Top-1 Accuracy**: 0.6190
- **Top-3 Accuracy**: 0.8509
- **Top-5 Accuracy**: 0.9181

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Azadirachta indica | 0.709 | 0.630 | 0.811 | 344 |
| Ailanthus excelsa | 0.680 | 0.713 | 0.649 | 134 |
| Vachellia nilotica | 0.671 | 0.684 | 0.658 | 257 |
| Cordia myxa | 0.667 | 1.000 | 0.500 | 2 |
| Phoenix sylvestris | 0.667 | 0.750 | 0.600 | 5 |
| prosopis juliflora | 0.632 | 0.552 | 0.740 | 50 |
| saraca asoca | 0.623 | 0.760 | 0.528 | 36 |
| Eucalyptus tereticornis | 0.571 | 1.000 | 0.400 | 5 |
| Pongamia pinnata | 0.516 | 0.471 | 0.571 | 14 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia procera | 0.364 | 0.667 | 0.250 | 8 |
| Albizia lebbeck | 0.200 | 0.200 | 0.200 | 5 |
| Vachellia leucophloea | 0.167 | 0.250 | 0.125 | 8 |
| Ziziphus jujuba | 0.143 | 0.500 | 0.083 | 12 |
| Moringa oleifera | 0.133 | 0.167 | 0.111 | 9 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Mangifera indica | 0.000 | 0.000 | 0.000 | 13 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 442
- Error rate: 0.3810
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Azadirachta indica | 31 |
| Vachellia nilotica | Azadirachta indica | 31 |
| Ailanthus excelsa | Azadirachta indica | 28 |
| Prosopis cineraria | Vachellia nilotica | 28 |
| Vachellia nilotica | Prosopis cineraria | 24 |
| Azadirachta indica | Vachellia nilotica | 19 |
| Vachellia nilotica | prosopis juliflora | 17 |
| Azadirachta indica | Prosopis cineraria | 14 |
| saraca asoca | Azadirachta indica | 12 |
| Dalbergia sissoo | Azadirachta indica | 11 |