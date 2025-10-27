# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-23 03:09:13

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6398
- **Balanced Accuracy**: 0.4118
- **F1 Score (Macro)**: 0.3899
- **F1 Score (Weighted)**: 0.6327
- **Precision (Macro)**: 0.4106
- **Recall (Macro)**: 0.4118
- **Top-1 Accuracy**: 0.6398
- **Top-3 Accuracy**: 0.8526
- **Top-5 Accuracy**: 0.9228

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Eucalyptus tereticornis | 0.800 | 1.000 | 0.667 | 3 |
| prosopis juliflora | 0.750 | 0.632 | 0.923 | 39 |
| Vachellia nilotica | 0.723 | 0.754 | 0.695 | 190 |
| Ailanthus excelsa | 0.692 | 0.589 | 0.838 | 99 |
| saraca asoca | 0.688 | 0.629 | 0.759 | 29 |
| Azadirachta indica | 0.674 | 0.748 | 0.614 | 251 |
| Ficus benghalensis | 0.667 | 0.667 | 0.667 | 12 |
| Prosopis cineraria | 0.643 | 0.600 | 0.692 | 91 |
| Morus Alba | 0.605 | 0.591 | 0.619 | 21 |
| Aegle Marmelos | 0.545 | 0.545 | 0.545 | 11 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia procera | 0.267 | 0.286 | 0.250 | 8 |
| Albizia lebbeck | 0.250 | 0.333 | 0.200 | 5 |
| Ficus religiosa | 0.235 | 0.333 | 0.182 | 11 |
| Phoenix sylvestris | 0.222 | 0.200 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 8 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 308
- Error rate: 0.3602
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 28 |
| Azadirachta indica | Vachellia nilotica | 18 |
| Azadirachta indica | Prosopis cineraria | 14 |
| Vachellia nilotica | Prosopis cineraria | 14 |
| Vachellia nilotica | Azadirachta indica | 14 |
| Vachellia nilotica | prosopis juliflora | 10 |
| Prosopis cineraria | Vachellia nilotica | 9 |
| Prosopis cineraria | Azadirachta indica | 8 |
| Vachellia nilotica | Ailanthus excelsa | 7 |
| Azadirachta indica | prosopis juliflora | 7 |