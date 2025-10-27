# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-30 02:21:32

## Dataset Summary
- Total test samples: 662
- Number of species classes: 197

## Overall Performance Metrics
- **Accuracy**: 0.1737
- **Balanced Accuracy**: 0.0583
- **F1 Score (Macro)**: 0.0206
- **F1 Score (Weighted)**: 0.1935
- **Precision (Macro)**: 0.0323
- **Recall (Macro)**: 0.0213
- **Top-1 Accuracy**: 0.1722
- **Top-3 Accuracy**: 0.3414
- **Top-5 Accuracy**: 0.4199

### Confidence Statistics
- Mean: 0.0058
- Std: 0.0000
- Range: [0.0056, 0.0063]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Azadirachta indica | 0.415 | 0.413 | 0.416 | 178 |
| prosopis juliflora | 0.304 | 0.318 | 0.292 | 24 |
| Vachellia nilotica | 0.265 | 0.533 | 0.176 | 136 |
| Moringa oleifera | 0.176 | 0.115 | 0.375 | 8 |
| Dalbergia sissoo | 0.125 | 0.333 | 0.077 | 26 |
| Ficus religiosa | 0.118 | 0.125 | 0.111 | 9 |
| Morus Alba | 0.070 | 0.054 | 0.100 | 20 |
| Ailanthus excelsa | 0.051 | 0.500 | 0.027 | 74 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 4 |
| Acrocarpus fraxinifolius | 0.000 | 0.000 | 0.000 | 0 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Pongamia pinnata | 0.000 | 0.000 | 0.000 | 13 |
| Prosopis cineraria | 0.000 | 0.000 | 0.000 | 81 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 3 |
| Ziziphus jujuba | 0.000 | 0.000 | 0.000 | 6 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 1 |
| Aegle Marmelos | 0.000 | 0.000 | 0.000 | 14 |
| saraca asoca | 0.000 | 0.000 | 0.000 | 26 |
| Eucalyptus tereticornis | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 547
- Error rate: 0.8263
- Mean confidence on errors: 0.0058
- Mean confidence on correct: 0.0059

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Vachellia nilotica | Senegalia senegal | 35 |
| Vachellia nilotica | Azadirachta indica | 22 |
| Prosopis cineraria | Azadirachta indica | 20 |
| Prosopis cineraria | Senegalia senegal | 18 |
| Ailanthus excelsa | Azadirachta indica | 18 |
| Ailanthus excelsa | Senegalia senegal | 15 |
| saraca asoca | Azadirachta indica | 13 |
| Azadirachta indica | Senegalia senegal | 13 |
| Azadirachta indica | Morus Alba | 10 |
| Azadirachta indica | Boswellia serrata | 8 |