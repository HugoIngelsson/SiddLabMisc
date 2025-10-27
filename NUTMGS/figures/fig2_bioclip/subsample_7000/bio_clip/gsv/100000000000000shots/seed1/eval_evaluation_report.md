# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 22:51:43

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6543
- **Balanced Accuracy**: 0.3969
- **F1 Score (Macro)**: 0.4258
- **F1 Score (Weighted)**: 0.6417
- **Precision (Macro)**: 0.4844
- **Recall (Macro)**: 0.3969
- **Top-1 Accuracy**: 0.6534
- **Top-3 Accuracy**: 0.8724
- **Top-5 Accuracy**: 0.9293

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0053, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Phoenix sylvestris | 0.800 | 0.800 | 0.800 | 5 |
| Eucalyptus tereticornis | 0.750 | 1.000 | 0.600 | 5 |
| Azadirachta indica | 0.744 | 0.681 | 0.820 | 344 |
| Ailanthus excelsa | 0.697 | 0.708 | 0.687 | 134 |
| Vachellia nilotica | 0.690 | 0.667 | 0.716 | 257 |
| prosopis juliflora | 0.685 | 0.638 | 0.740 | 50 |
| Morus Alba | 0.667 | 0.889 | 0.533 | 30 |
| saraca asoca | 0.636 | 0.700 | 0.583 | 36 |
| Pongamia pinnata | 0.615 | 0.667 | 0.571 | 14 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Ziziphus jujuba | 0.250 | 0.500 | 0.167 | 12 |
| Albizia procera | 0.235 | 0.222 | 0.250 | 8 |
| Mangifera indica | 0.211 | 0.333 | 0.154 | 13 |
| Moringa oleifera | 0.167 | 0.333 | 0.111 | 9 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 8 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 401
- Error rate: 0.3457
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 30 |
| Vachellia nilotica | Prosopis cineraria | 23 |
| Prosopis cineraria | Azadirachta indica | 23 |
| Vachellia nilotica | Azadirachta indica | 22 |
| Ailanthus excelsa | Azadirachta indica | 20 |
| Azadirachta indica | Vachellia nilotica | 18 |
| Azadirachta indica | Prosopis cineraria | 15 |
| Ailanthus excelsa | Vachellia nilotica | 14 |
| Azadirachta indica | Ailanthus excelsa | 13 |
| Vachellia nilotica | prosopis juliflora | 13 |