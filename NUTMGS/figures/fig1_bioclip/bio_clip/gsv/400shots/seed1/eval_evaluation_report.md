# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 22:23:34

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6094
- **Balanced Accuracy**: 0.4343
- **F1 Score (Macro)**: 0.4065
- **F1 Score (Weighted)**: 0.6105
- **Precision (Macro)**: 0.4031
- **Recall (Macro)**: 0.4343
- **Top-1 Accuracy**: 0.6058
- **Top-3 Accuracy**: 0.8538
- **Top-5 Accuracy**: 0.9205

### Confidence Statistics
- Mean: 0.0056
- Std: 0.0000
- Range: [0.0054, 0.0060]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Eucalyptus tereticornis | 0.800 | 1.000 | 0.667 | 3 |
| saraca asoca | 0.698 | 0.647 | 0.759 | 29 |
| Vachellia nilotica | 0.692 | 0.781 | 0.621 | 190 |
| Ailanthus excelsa | 0.690 | 0.608 | 0.798 | 99 |
| prosopis juliflora | 0.680 | 0.547 | 0.897 | 39 |
| Cassia fistula | 0.667 | 0.500 | 1.000 | 1 |
| Azadirachta indica | 0.640 | 0.730 | 0.570 | 251 |
| Ficus benghalensis | 0.609 | 0.636 | 0.583 | 12 |
| Prosopis cineraria | 0.594 | 0.541 | 0.659 | 91 |
| Pongamia pinnata | 0.583 | 0.636 | 0.538 | 13 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Ziziphus jujuba | 0.308 | 0.400 | 0.250 | 8 |
| Albizia procera | 0.286 | 0.333 | 0.250 | 8 |
| Moringa oleifera | 0.286 | 0.231 | 0.375 | 8 |
| Ficus religiosa | 0.261 | 0.250 | 0.273 | 11 |
| Phoenix sylvestris | 0.222 | 0.200 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 334
- Error rate: 0.3906
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 29 |
| Vachellia nilotica | Azadirachta indica | 19 |
| Azadirachta indica | Prosopis cineraria | 17 |
| Vachellia nilotica | Prosopis cineraria | 14 |
| Vachellia nilotica | prosopis juliflora | 13 |
| Azadirachta indica | Vachellia nilotica | 11 |
| Prosopis cineraria | Azadirachta indica | 10 |
| Prosopis cineraria | Vachellia nilotica | 9 |
| Azadirachta indica | prosopis juliflora | 9 |
| Vachellia nilotica | Ailanthus excelsa | 8 |