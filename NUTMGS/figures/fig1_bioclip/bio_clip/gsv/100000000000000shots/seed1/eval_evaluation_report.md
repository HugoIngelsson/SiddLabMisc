# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-22 00:02:17

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6655
- **Balanced Accuracy**: 0.3911
- **F1 Score (Macro)**: 0.3963
- **F1 Score (Weighted)**: 0.6518
- **Precision (Macro)**: 0.4427
- **Recall (Macro)**: 0.3911
- **Top-1 Accuracy**: 0.6643
- **Top-3 Accuracy**: 0.8784
- **Top-5 Accuracy**: 0.9287

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0061]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Ailanthus excelsa | 0.761 | 0.736 | 0.788 | 99 |
| Vachellia nilotica | 0.729 | 0.722 | 0.737 | 190 |
| saraca asoca | 0.724 | 0.724 | 0.724 | 29 |
| Azadirachta indica | 0.715 | 0.664 | 0.773 | 251 |
| Cassia fistula | 0.667 | 0.500 | 1.000 | 1 |
| prosopis juliflora | 0.667 | 0.643 | 0.692 | 39 |
| Eucalyptus tereticornis | 0.667 | 0.667 | 0.667 | 3 |
| Ficus benghalensis | 0.640 | 0.615 | 0.667 | 12 |
| Prosopis cineraria | 0.621 | 0.640 | 0.604 | 91 |
| Pongamia pinnata | 0.615 | 0.615 | 0.615 | 13 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Dalbergia sissoo | 0.300 | 0.462 | 0.222 | 27 |
| Moringa oleifera | 0.286 | 0.333 | 0.250 | 8 |
| Phoenix sylvestris | 0.222 | 0.200 | 0.250 | 4 |
| Ziziphus jujuba | 0.182 | 0.333 | 0.125 | 8 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 3 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 286
- Error rate: 0.3345
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Vachellia nilotica | Azadirachta indica | 21 |
| Azadirachta indica | Vachellia nilotica | 16 |
| Prosopis cineraria | Azadirachta indica | 15 |
| Azadirachta indica | Ailanthus excelsa | 14 |
| Prosopis cineraria | Vachellia nilotica | 13 |
| Dalbergia sissoo | Azadirachta indica | 12 |
| Ailanthus excelsa | Azadirachta indica | 12 |
| prosopis juliflora | Vachellia nilotica | 10 |
| Azadirachta indica | Prosopis cineraria | 10 |
| Vachellia nilotica | Prosopis cineraria | 10 |