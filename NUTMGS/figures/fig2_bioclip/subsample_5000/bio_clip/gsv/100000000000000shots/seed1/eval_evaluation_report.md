# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-24 17:31:17

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6388
- **Balanced Accuracy**: 0.3765
- **F1 Score (Macro)**: 0.3817
- **F1 Score (Weighted)**: 0.6249
- **Precision (Macro)**: 0.4201
- **Recall (Macro)**: 0.3626
- **Top-1 Accuracy**: 0.6388
- **Top-3 Accuracy**: 0.8698
- **Top-5 Accuracy**: 0.9284

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Eucalyptus tereticornis | 0.750 | 1.000 | 0.600 | 5 |
| Azadirachta indica | 0.731 | 0.666 | 0.811 | 344 |
| Vachellia nilotica | 0.708 | 0.690 | 0.728 | 257 |
| Ailanthus excelsa | 0.700 | 0.713 | 0.687 | 134 |
| prosopis juliflora | 0.648 | 0.618 | 0.680 | 50 |
| saraca asoca | 0.646 | 0.724 | 0.583 | 36 |
| Phoenix sylvestris | 0.600 | 0.600 | 0.600 | 5 |
| Pongamia pinnata | 0.516 | 0.471 | 0.571 | 14 |
| Prosopis cineraria | 0.492 | 0.545 | 0.448 | 134 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Vachellia leucophloea | 0.333 | 0.500 | 0.250 | 8 |
| Ziziphus jujuba | 0.235 | 0.400 | 0.167 | 12 |
| Mangifera indica | 0.200 | 0.286 | 0.154 | 13 |
| Moringa oleifera | 0.154 | 0.250 | 0.111 | 9 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 419
- Error rate: 0.3612
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 29 |
| Ailanthus excelsa | Azadirachta indica | 24 |
| Prosopis cineraria | Azadirachta indica | 24 |
| Vachellia nilotica | Azadirachta indica | 24 |
| Vachellia nilotica | Prosopis cineraria | 17 |
| Azadirachta indica | Vachellia nilotica | 17 |
| Azadirachta indica | Prosopis cineraria | 16 |
| Ailanthus excelsa | Vachellia nilotica | 12 |
| Vachellia nilotica | prosopis juliflora | 12 |
| Azadirachta indica | Ailanthus excelsa | 11 |