# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 21:06:13

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6035
- **Balanced Accuracy**: 0.4348
- **F1 Score (Macro)**: 0.3908
- **F1 Score (Weighted)**: 0.6098
- **Precision (Macro)**: 0.3724
- **Recall (Macro)**: 0.4348
- **Top-1 Accuracy**: 0.6012
- **Top-3 Accuracy**: 0.8292
- **Top-5 Accuracy**: 0.9076

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0060]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Eucalyptus tereticornis | 0.857 | 0.750 | 1.000 | 3 |
| Ailanthus excelsa | 0.719 | 0.629 | 0.838 | 99 |
| saraca asoca | 0.688 | 0.629 | 0.759 | 29 |
| Vachellia nilotica | 0.669 | 0.739 | 0.611 | 190 |
| Cassia fistula | 0.667 | 0.500 | 1.000 | 1 |
| Azadirachta indica | 0.665 | 0.799 | 0.570 | 251 |
| prosopis juliflora | 0.626 | 0.517 | 0.795 | 39 |
| Ficus benghalensis | 0.615 | 0.571 | 0.667 | 12 |
| Prosopis cineraria | 0.594 | 0.564 | 0.626 | 91 |
| Morus Alba | 0.537 | 0.550 | 0.524 | 21 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Dalbergia sissoo | 0.308 | 0.263 | 0.370 | 27 |
| Mangifera indica | 0.300 | 0.300 | 0.300 | 10 |
| Albizia lebbeck | 0.200 | 0.200 | 0.200 | 5 |
| Moringa oleifera | 0.182 | 0.143 | 0.250 | 8 |
| Phoenix sylvestris | 0.182 | 0.143 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 339
- Error rate: 0.3965
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 23 |
| Azadirachta indica | Prosopis cineraria | 18 |
| Vachellia nilotica | Azadirachta indica | 17 |
| Vachellia nilotica | Prosopis cineraria | 14 |
| Vachellia nilotica | prosopis juliflora | 13 |
| Prosopis cineraria | Vachellia nilotica | 13 |
| Azadirachta indica | Vachellia nilotica | 10 |
| Azadirachta indica | Dalbergia sissoo | 9 |
| Vachellia nilotica | Ailanthus excelsa | 8 |
| Azadirachta indica | Aegle Marmelos | 7 |