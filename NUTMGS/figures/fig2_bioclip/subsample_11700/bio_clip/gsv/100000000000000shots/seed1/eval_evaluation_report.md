# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-30 19:09:30

## Dataset Summary
- Total test samples: 1465
- Number of species classes: 223

## Overall Performance Metrics
- **Accuracy**: 0.6990
- **Balanced Accuracy**: 0.3231
- **F1 Score (Macro)**: 0.3235
- **F1 Score (Weighted)**: 0.6839
- **Precision (Macro)**: 0.3469
- **Recall (Macro)**: 0.3119
- **Top-1 Accuracy**: 0.7010
- **Top-3 Accuracy**: 0.9038
- **Top-5 Accuracy**: 0.9461

### Confidence Statistics
- Mean: 0.0054
- Std: 0.0000
- Range: [0.0051, 0.0057]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Vachellia nilotica | 0.782 | 0.739 | 0.832 | 333 |
| Phoenix sylvestris | 0.778 | 0.778 | 0.778 | 9 |
| prosopis juliflora | 0.767 | 0.750 | 0.785 | 65 |
| Azadirachta indica | 0.763 | 0.710 | 0.824 | 488 |
| saraca asoca | 0.730 | 0.750 | 0.711 | 38 |
| Ailanthus excelsa | 0.691 | 0.721 | 0.664 | 140 |
| Prosopis cineraria | 0.652 | 0.732 | 0.588 | 153 |
| Pongamia pinnata | 0.600 | 0.600 | 0.600 | 15 |
| Ficus benghalensis | 0.571 | 0.556 | 0.588 | 17 |
| Eucalyptus tereticornis | 0.556 | 0.556 | 0.556 | 9 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia lebbeck | 0.000 | 0.000 | 0.000 | 5 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Butea monosperma | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 2 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Moringa oleifera | 0.000 | 0.000 | 0.000 | 9 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 9 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 441
- Error rate: 0.3010
- Mean confidence on errors: 0.0053
- Mean confidence on correct: 0.0054

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 32 |
| Azadirachta indica | Vachellia nilotica | 31 |
| Ailanthus excelsa | Azadirachta indica | 30 |
| Vachellia nilotica | Azadirachta indica | 27 |
| Prosopis cineraria | Azadirachta indica | 22 |
| Dalbergia sissoo | Azadirachta indica | 18 |
| Azadirachta indica | Prosopis cineraria | 12 |
| Mangifera indica | Azadirachta indica | 11 |
| Azadirachta indica | Ailanthus excelsa | 11 |
| prosopis juliflora | Vachellia nilotica | 10 |