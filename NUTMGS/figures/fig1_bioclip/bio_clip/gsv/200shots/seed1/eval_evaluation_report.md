# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 19:46:38

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.5696
- **Balanced Accuracy**: 0.4418
- **F1 Score (Macro)**: 0.3771
- **F1 Score (Weighted)**: 0.5788
- **Precision (Macro)**: 0.3551
- **Recall (Macro)**: 0.4418
- **Top-1 Accuracy**: 0.5673
- **Top-3 Accuracy**: 0.8129
- **Top-5 Accuracy**: 0.8959

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0060]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| saraca asoca | 0.697 | 0.622 | 0.793 | 29 |
| Ailanthus excelsa | 0.676 | 0.640 | 0.717 | 99 |
| Cassia fistula | 0.667 | 0.500 | 1.000 | 1 |
| Eucalyptus tereticornis | 0.667 | 0.500 | 1.000 | 3 |
| prosopis juliflora | 0.654 | 0.523 | 0.872 | 39 |
| Ficus benghalensis | 0.640 | 0.615 | 0.667 | 12 |
| Vachellia nilotica | 0.634 | 0.763 | 0.542 | 190 |
| Azadirachta indica | 0.630 | 0.778 | 0.530 | 251 |
| Prosopis cineraria | 0.516 | 0.451 | 0.604 | 91 |
| Morus Alba | 0.511 | 0.462 | 0.571 | 21 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Mangifera indica | 0.300 | 0.300 | 0.300 | 10 |
| Moringa oleifera | 0.286 | 0.231 | 0.375 | 8 |
| Phoenix sylvestris | 0.250 | 0.167 | 0.500 | 4 |
| Albizia lebbeck | 0.222 | 0.250 | 0.200 | 5 |
| Ziziphus jujuba | 0.182 | 0.333 | 0.125 | 8 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 368
- Error rate: 0.4304
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Prosopis cineraria | 24 |
| Azadirachta indica | Ailanthus excelsa | 22 |
| Vachellia nilotica | Prosopis cineraria | 21 |
| Vachellia nilotica | Azadirachta indica | 18 |
| Prosopis cineraria | Vachellia nilotica | 15 |
| Vachellia nilotica | prosopis juliflora | 15 |
| Azadirachta indica | Dalbergia sissoo | 14 |
| Azadirachta indica | Aegle Marmelos | 8 |
| Azadirachta indica | prosopis juliflora | 8 |
| Vachellia nilotica | Vachellia leucophloea | 7 |