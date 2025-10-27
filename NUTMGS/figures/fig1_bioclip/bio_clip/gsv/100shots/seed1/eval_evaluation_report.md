# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 18:48:43

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.5205
- **Balanced Accuracy**: 0.4501
- **F1 Score (Macro)**: 0.3345
- **F1 Score (Weighted)**: 0.5466
- **Precision (Macro)**: 0.2995
- **Recall (Macro)**: 0.4501
- **Top-1 Accuracy**: 0.5181
- **Top-3 Accuracy**: 0.7637
- **Top-5 Accuracy**: 0.8526

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0053, 0.0059]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| prosopis juliflora | 0.652 | 0.566 | 0.769 | 39 |
| Vachellia nilotica | 0.643 | 0.815 | 0.532 | 190 |
| saraca asoca | 0.635 | 0.588 | 0.690 | 29 |
| Ailanthus excelsa | 0.618 | 0.568 | 0.677 | 99 |
| Azadirachta indica | 0.561 | 0.854 | 0.418 | 251 |
| Prosopis cineraria | 0.529 | 0.510 | 0.549 | 91 |
| Cassia fistula | 0.500 | 0.333 | 1.000 | 1 |
| Albizia lebbeck | 0.429 | 0.333 | 0.600 | 5 |
| Ficus religiosa | 0.429 | 0.353 | 0.545 | 11 |
| Ficus benghalensis | 0.414 | 0.353 | 0.500 | 12 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Aegle Marmelos | 0.308 | 0.214 | 0.545 | 11 |
| Mangifera indica | 0.242 | 0.174 | 0.400 | 10 |
| Acalypha fruticosa | 0.222 | 0.167 | 0.333 | 3 |
| Moringa oleifera | 0.163 | 0.098 | 0.500 | 8 |
| Phoenix sylvestris | 0.133 | 0.091 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 410
- Error rate: 0.4795
- Mean confidence on errors: 0.0055
- Mean confidence on correct: 0.0056

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 25 |
| Azadirachta indica | Prosopis cineraria | 21 |
| Azadirachta indica | Moringa oleifera | 20 |
| Vachellia nilotica | Prosopis cineraria | 19 |
| Vachellia nilotica | prosopis juliflora | 11 |
| Prosopis cineraria | Ailanthus excelsa | 8 |
| Azadirachta indica | Mangifera indica | 8 |
| Vachellia nilotica | Ailanthus excelsa | 7 |
| Azadirachta indica | Albizia procera | 7 |
| Prosopis cineraria | Vachellia nilotica | 7 |