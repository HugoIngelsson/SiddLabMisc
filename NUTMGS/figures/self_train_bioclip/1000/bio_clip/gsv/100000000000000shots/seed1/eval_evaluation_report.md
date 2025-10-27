# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-08-01 00:49:28

## Dataset Summary
- Total test samples: 1398
- Number of species classes: 236

## Overall Performance Metrics
- **Accuracy**: 0.4356
- **Balanced Accuracy**: 0.2168
- **F1 Score (Macro)**: 0.1717
- **F1 Score (Weighted)**: 0.4481
- **Precision (Macro)**: 0.2552
- **Recall (Macro)**: 0.2093
- **Top-1 Accuracy**: 0.4356
- **Top-3 Accuracy**: 0.7425
- **Top-5 Accuracy**: 0.8476

### Confidence Statistics
- Mean: 0.0051
- Std: 0.0000
- Range: [0.0048, 0.0056]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Phoenix sylvestris | 0.714 | 0.714 | 0.714 | 7 |
| Vachellia nilotica | 0.631 | 0.627 | 0.636 | 319 |
| Ailanthus excelsa | 0.528 | 0.474 | 0.596 | 136 |
| Azadirachta indica | 0.507 | 0.748 | 0.384 | 456 |
| saraca asoca | 0.431 | 0.786 | 0.297 | 37 |
| Prosopis cineraria | 0.417 | 0.326 | 0.578 | 147 |
| Morus Alba | 0.286 | 0.256 | 0.324 | 34 |
| Ficus benghalensis | 0.273 | 0.600 | 0.176 | 17 |
| Ficus religiosa | 0.267 | 0.444 | 0.190 | 21 |
| Pongamia pinnata | 0.211 | 0.500 | 0.133 | 15 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Butea monosperma | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 2 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Moringa oleifera | 0.000 | 0.000 | 0.000 | 9 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 9 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 8 |

## Error Analysis Summary
- Total errors: 789
- Error rate: 0.5644
- Mean confidence on errors: 0.0051
- Mean confidence on correct: 0.0052

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Vachellia nilotica | Prosopis cineraria | 70 |
| Azadirachta indica | Prosopis cineraria | 55 |
| Azadirachta indica | Ailanthus excelsa | 54 |
| Azadirachta indica | Mangifera indica | 46 |
| Azadirachta indica | Vachellia nilotica | 40 |
| prosopis juliflora | Prosopis cineraria | 34 |
| Prosopis cineraria | Vachellia nilotica | 27 |
| Azadirachta indica | Albizia lebbeck | 25 |
| prosopis juliflora | Vachellia nilotica | 23 |
| Azadirachta indica | Ziziphus jujuba | 23 |