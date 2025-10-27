# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 11:08:10

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.5526
- **Balanced Accuracy**: 0.2750
- **F1 Score (Macro)**: 0.2780
- **F1 Score (Weighted)**: 0.5382
- **Precision (Macro)**: 0.3343
- **Recall (Macro)**: 0.2648
- **Top-1 Accuracy**: 0.5509
- **Top-3 Accuracy**: 0.8103
- **Top-5 Accuracy**: 0.8957

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0060]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Azadirachta indica | 0.668 | 0.611 | 0.738 | 344 |
| Vachellia nilotica | 0.633 | 0.613 | 0.654 | 257 |
| Ailanthus excelsa | 0.556 | 0.576 | 0.537 | 134 |
| prosopis juliflora | 0.554 | 0.549 | 0.560 | 50 |
| saraca asoca | 0.525 | 0.640 | 0.444 | 36 |
| Phoenix sylvestris | 0.444 | 0.500 | 0.400 | 5 |
| Prosopis cineraria | 0.432 | 0.466 | 0.403 | 134 |
| Ficus benghalensis | 0.385 | 0.455 | 0.333 | 15 |
| Dalbergia sissoo | 0.381 | 0.364 | 0.400 | 40 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Ziziphus jujuba | 0.154 | 1.000 | 0.083 | 12 |
| Mangifera indica | 0.091 | 0.111 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Moringa oleifera | 0.000 | 0.000 | 0.000 | 9 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 5 |

## Error Analysis Summary
- Total errors: 519
- Error rate: 0.4474
- Mean confidence on errors: 0.0055
- Mean confidence on correct: 0.0056

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 38 |
| Ailanthus excelsa | Azadirachta indica | 35 |
| Vachellia nilotica | Azadirachta indica | 34 |
| Prosopis cineraria | Azadirachta indica | 30 |
| Vachellia nilotica | Prosopis cineraria | 27 |
| Azadirachta indica | Vachellia nilotica | 21 |
| Azadirachta indica | Prosopis cineraria | 19 |
| Azadirachta indica | Ailanthus excelsa | 17 |
| Ailanthus excelsa | Vachellia nilotica | 14 |
| prosopis juliflora | Vachellia nilotica | 13 |