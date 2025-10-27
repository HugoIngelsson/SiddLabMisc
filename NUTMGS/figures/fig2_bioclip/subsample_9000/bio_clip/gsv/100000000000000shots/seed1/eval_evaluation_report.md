# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-25 05:42:19

## Dataset Summary
- Total test samples: 1160
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6741
- **Balanced Accuracy**: 0.4158
- **F1 Score (Macro)**: 0.4229
- **F1 Score (Weighted)**: 0.6639
- **Precision (Macro)**: 0.4852
- **Recall (Macro)**: 0.4004
- **Top-1 Accuracy**: 0.6707
- **Top-3 Accuracy**: 0.8819
- **Top-5 Accuracy**: 0.9405

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Cassia fistula | 1.000 | 1.000 | 1.000 | 1 |
| Phoenix sylvestris | 0.800 | 0.800 | 0.800 | 5 |
| Azadirachta indica | 0.767 | 0.714 | 0.828 | 344 |
| Ailanthus excelsa | 0.732 | 0.740 | 0.724 | 134 |
| Vachellia nilotica | 0.711 | 0.695 | 0.728 | 257 |
| saraca asoca | 0.667 | 0.733 | 0.611 | 36 |
| prosopis juliflora | 0.660 | 0.642 | 0.680 | 50 |
| Morus Alba | 0.607 | 0.654 | 0.567 | 30 |
| Ficus benghalensis | 0.593 | 0.667 | 0.533 | 15 |
| Albizia lebbeck | 0.571 | 1.000 | 0.400 | 5 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia procera | 0.364 | 0.286 | 0.500 | 8 |
| Ziziphus jujuba | 0.353 | 0.600 | 0.250 | 12 |
| Vachellia leucophloea | 0.182 | 0.333 | 0.125 | 8 |
| Moringa oleifera | 0.167 | 0.333 | 0.111 | 9 |
| Mangifera indica | 0.118 | 0.250 | 0.077 | 13 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 378
- Error rate: 0.3259
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 28 |
| Vachellia nilotica | Prosopis cineraria | 21 |
| Vachellia nilotica | Azadirachta indica | 20 |
| Prosopis cineraria | Azadirachta indica | 19 |
| Ailanthus excelsa | Azadirachta indica | 17 |
| Azadirachta indica | Vachellia nilotica | 16 |
| Ailanthus excelsa | Vachellia nilotica | 13 |
| Azadirachta indica | Ailanthus excelsa | 12 |
| Vachellia nilotica | prosopis juliflora | 11 |
| Azadirachta indica | Prosopis cineraria | 11 |