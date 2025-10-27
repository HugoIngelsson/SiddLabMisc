# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-23 01:26:10

## Dataset Summary
- Total test samples: 855
- Number of species classes: 210

## Overall Performance Metrics
- **Accuracy**: 0.6421
- **Balanced Accuracy**: 0.4317
- **F1 Score (Macro)**: 0.4025
- **F1 Score (Weighted)**: 0.6349
- **Precision (Macro)**: 0.4042
- **Recall (Macro)**: 0.4317
- **Top-1 Accuracy**: 0.6386
- **Top-3 Accuracy**: 0.8526
- **Top-5 Accuracy**: 0.9181

### Confidence Statistics
- Mean: 0.0057
- Std: 0.0000
- Range: [0.0054, 0.0060]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| saraca asoca | 0.738 | 0.667 | 0.828 | 29 |
| prosopis juliflora | 0.729 | 0.614 | 0.897 | 39 |
| Vachellia nilotica | 0.718 | 0.756 | 0.684 | 190 |
| Ailanthus excelsa | 0.706 | 0.610 | 0.838 | 99 |
| Azadirachta indica | 0.670 | 0.726 | 0.622 | 251 |
| Ficus benghalensis | 0.667 | 0.600 | 0.750 | 12 |
| Prosopis cineraria | 0.635 | 0.604 | 0.670 | 91 |
| Eucalyptus tereticornis | 0.571 | 0.500 | 0.667 | 3 |
| Aegle Marmelos | 0.522 | 0.500 | 0.545 | 11 |
| Morus Alba | 0.512 | 0.500 | 0.524 | 21 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Dalbergia sissoo | 0.314 | 0.333 | 0.296 | 27 |
| Ziziphus jujuba | 0.308 | 0.400 | 0.250 | 8 |
| Moringa oleifera | 0.300 | 0.250 | 0.375 | 8 |
| Ficus religiosa | 0.250 | 0.400 | 0.182 | 11 |
| Phoenix sylvestris | 0.222 | 0.200 | 0.250 | 4 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 5 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 306
- Error rate: 0.3579
- Mean confidence on errors: 0.0056
- Mean confidence on correct: 0.0057

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Azadirachta indica | Ailanthus excelsa | 24 |
| Azadirachta indica | Vachellia nilotica | 21 |
| Vachellia nilotica | Azadirachta indica | 18 |
| Azadirachta indica | Prosopis cineraria | 14 |
| Vachellia nilotica | Prosopis cineraria | 12 |
| Vachellia nilotica | prosopis juliflora | 12 |
| Prosopis cineraria | Azadirachta indica | 10 |
| Azadirachta indica | saraca asoca | 8 |
| Dalbergia sissoo | Azadirachta indica | 7 |
| Prosopis cineraria | Vachellia nilotica | 7 |