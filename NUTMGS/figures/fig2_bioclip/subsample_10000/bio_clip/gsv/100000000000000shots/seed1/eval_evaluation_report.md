# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-29 18:30:36

## Dataset Summary
- Total test samples: 1126
- Number of species classes: 223

## Overall Performance Metrics
- **Accuracy**: 0.7007
- **Balanced Accuracy**: 0.4132
- **F1 Score (Macro)**: 0.4388
- **F1 Score (Weighted)**: 0.6887
- **Precision (Macro)**: 0.4827
- **Recall (Macro)**: 0.4132
- **Top-1 Accuracy**: 0.7007
- **Top-3 Accuracy**: 0.9023
- **Top-5 Accuracy**: 0.9547

### Confidence Statistics
- Mean: 0.0054
- Std: 0.0000
- Range: [0.0051, 0.0058]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Alstonia scholaris | 1.000 | 1.000 | 1.000 | 1 |
| Azadirachta indica | 0.764 | 0.707 | 0.831 | 354 |
| Vachellia nilotica | 0.755 | 0.767 | 0.743 | 253 |
| Ailanthus excelsa | 0.738 | 0.732 | 0.744 | 121 |
| Ficus benghalensis | 0.714 | 0.714 | 0.714 | 14 |
| saraca asoca | 0.687 | 0.657 | 0.719 | 32 |
| prosopis juliflora | 0.686 | 0.655 | 0.720 | 50 |
| Phoenix sylvestris | 0.667 | 0.800 | 0.571 | 7 |
| Prosopis cineraria | 0.639 | 0.647 | 0.631 | 122 |
| Pongamia pinnata | 0.636 | 0.778 | 0.538 | 13 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Albizia lebbeck | 0.333 | 0.500 | 0.250 | 4 |
| Mangifera indica | 0.222 | 0.400 | 0.154 | 13 |
| Aegle Marmelos | 0.182 | 0.250 | 0.143 | 14 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Butea monosperma | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 7 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 337
- Error rate: 0.2993
- Mean confidence on errors: 0.0053
- Mean confidence on correct: 0.0054

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Vachellia nilotica | Azadirachta indica | 23 |
| Ailanthus excelsa | Azadirachta indica | 18 |
| Prosopis cineraria | Vachellia nilotica | 17 |
| Vachellia nilotica | Prosopis cineraria | 17 |
| Prosopis cineraria | Azadirachta indica | 17 |
| Azadirachta indica | Vachellia nilotica | 16 |
| Azadirachta indica | Ailanthus excelsa | 12 |
| Vachellia nilotica | prosopis juliflora | 11 |
| Azadirachta indica | Prosopis cineraria | 10 |
| Dalbergia sissoo | Azadirachta indica | 10 |