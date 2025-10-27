# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 12:09:22

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.5707
- **Balanced Accuracy**: 0.3061
- **F1 Score (Macro)**: 0.3153
- **F1 Score (Weighted)**: 0.5562
- **Precision (Macro)**: 0.3624
- **Recall (Macro)**: 0.3061
- **Top-1 Accuracy**: 0.5724
- **Top-3 Accuracy**: 0.8414
- **Top-5 Accuracy**: 0.9086

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Azadirachta indica | 0.671 | 0.611 | 0.744 | 344 |
| Vachellia nilotica | 0.658 | 0.650 | 0.665 | 257 |
| prosopis juliflora | 0.629 | 0.600 | 0.660 | 50 |
| Phoenix sylvestris | 0.600 | 0.600 | 0.600 | 5 |
| saraca asoca | 0.576 | 0.739 | 0.472 | 36 |
| Ailanthus excelsa | 0.576 | 0.602 | 0.552 | 134 |
| Pongamia pinnata | 0.444 | 0.364 | 0.571 | 14 |
| Prosopis cineraria | 0.436 | 0.439 | 0.433 | 134 |
| Dalbergia sissoo | 0.394 | 0.452 | 0.350 | 40 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Vachellia leucophloea | 0.222 | 1.000 | 0.125 | 8 |
| Mangifera indica | 0.105 | 0.167 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Moringa oleifera | 0.000 | 0.000 | 0.000 | 9 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 12 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 498
- Error rate: 0.4293
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Ailanthus excelsa | Azadirachta indica | 36 |
| Prosopis cineraria | Azadirachta indica | 31 |
| Vachellia nilotica | Prosopis cineraria | 30 |
| Prosopis cineraria | Vachellia nilotica | 27 |
| Vachellia nilotica | Azadirachta indica | 27 |
| Azadirachta indica | Prosopis cineraria | 23 |
| Azadirachta indica | Ailanthus excelsa | 20 |
| Azadirachta indica | Vachellia nilotica | 18 |
| Vachellia nilotica | prosopis juliflora | 15 |
| Ailanthus excelsa | Vachellia nilotica | 14 |