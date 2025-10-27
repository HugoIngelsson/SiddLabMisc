# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 23:51:04

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6304
- **Balanced Accuracy**: 0.4324
- **F1 Score (Macro)**: 0.4042
- **F1 Score (Weighted)**: 0.6273
- **Precision (Macro)**: 0.4101
- **Recall (Macro)**: 0.4324
- **Top-1 Accuracy**: 0.6304
- **Top-3 Accuracy**: 0.8491
- **Top-5 Accuracy**: 0.9076

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0053, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Eucalyptus tereticornis | 0.800 | 1.000 | 0.667 | 3 |
| Vachellia nilotica | 0.722 | 0.784 | 0.668 | 190 |
| saraca asoca | 0.708 | 0.639 | 0.793 | 29 |
| Ailanthus excelsa | 0.686 | 0.586 | 0.828 | 99 |
| prosopis juliflora | 0.681 | 0.582 | 0.821 | 39 |
| Azadirachta indica | 0.656 | 0.734 | 0.594 | 251 |
| Ficus benghalensis | 0.640 | 0.615 | 0.667 | 12 |
| Prosopis cineraria | 0.614 | 0.559 | 0.681 | 91 |
| Morus Alba | 0.591 | 0.565 | 0.619 | 21 |
| Pongamia pinnata | 0.533 | 0.471 | 0.615 | 13 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Moringa oleifera | 0.353 | 0.333 | 0.375 | 8 |
| Mangifera indica | 0.333 | 0.375 | 0.300 | 10 |
| Albizia procera | 0.250 | 0.250 | 0.250 | 8 |
| Ficus religiosa | 0.222 | 0.286 | 0.182 | 11 |
| Phoenix sylvestris | 0.200 | 0.167 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 316
- Error rate: 0.3696
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 30 |
| Azadirachta indica | Prosopis cineraria | 21 |
| Vachellia nilotica | Azadirachta indica | 17 |
| Vachellia nilotica | Prosopis cineraria | 16 |
| Vachellia nilotica | prosopis juliflora | 12 |
| Azadirachta indica | Vachellia nilotica | 11 |
| Prosopis cineraria | Azadirachta indica | 9 |
| Vachellia nilotica | Ailanthus excelsa | 7 |
| Prosopis cineraria | Ailanthus excelsa | 7 |
| Ailanthus excelsa | Azadirachta indica | 6 |