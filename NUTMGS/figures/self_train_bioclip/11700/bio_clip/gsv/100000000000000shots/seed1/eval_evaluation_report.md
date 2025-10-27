# CLIP-LoRA Species Classification Evaluation Report

Generated: 2025-07-31 21:47:31

## Dataset Summary
- Total test samples: 1398
- Number of species classes: 249

## Overall Performance Metrics
- **Accuracy**: 0.6795
- **Balanced Accuracy**: 0.3587
- **F1 Score (Macro)**: 0.3382
- **F1 Score (Weighted)**: 0.6700
- **Precision (Macro)**: 0.3625
- **Recall (Macro)**: 0.3464
- **Top-1 Accuracy**: 0.6824
- **Top-3 Accuracy**: 0.8848
- **Top-5 Accuracy**: 0.9335

### Confidence Statistics
- Mean: 0.0049
- Std: 0.0000
- Range: [0.0045, 0.0053]

## Top 10 Performing Species

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| prosopis juliflora | 0.800 | 0.788 | 0.812 | 64 |
| Vachellia nilotica | 0.773 | 0.748 | 0.799 | 319 |
| Azadirachta indica | 0.751 | 0.716 | 0.789 | 456 |
| Phoenix sylvestris | 0.714 | 0.714 | 0.714 | 7 |
| saraca asoca | 0.693 | 0.684 | 0.703 | 37 |
| Butea monosperma | 0.667 | 0.500 | 1.000 | 1 |
| Ailanthus excelsa | 0.657 | 0.667 | 0.647 | 136 |
| Prosopis cineraria | 0.633 | 0.664 | 0.605 | 147 |
| Pongamia pinnata | 0.562 | 0.529 | 0.600 | 15 |
| Morus Alba | 0.528 | 0.737 | 0.412 | 34 |

## Bottom 10 Performing Species (with samples)

| Species | F1 Score | Precision | Recall | Support |
|---------|----------|-----------|---------|---------|
| Moringa oleifera | 0.182 | 0.500 | 0.111 | 9 |
| Acalypha fruticosa | 0.000 | 0.000 | 0.000 | 5 |
| Albizia procera | 0.000 | 0.000 | 0.000 | 9 |
| Alstonia scholaris | 0.000 | 0.000 | 0.000 | 1 |
| Cassia fistula | 0.000 | 0.000 | 0.000 | 2 |
| Cordia myxa | 0.000 | 0.000 | 0.000 | 2 |
| Prosopis farcta | 0.000 | 0.000 | 0.000 | 1 |
| Terminalia arjuna | 0.000 | 0.000 | 0.000 | 1 |
| Vachellia leucophloea | 0.000 | 0.000 | 0.000 | 9 |
| Ziziphus mauritiana | 0.000 | 0.000 | 0.000 | 2 |

## Error Analysis Summary
- Total errors: 448
- Error rate: 0.3205
- Mean confidence on errors: 0.0048
- Mean confidence on correct: 0.0049

### Top 10 Most Confused Species Pairs

| True Species | Predicted Species | Count |
|--------------|-------------------|-------|
| Prosopis cineraria | Vachellia nilotica | 28 |
| Ailanthus excelsa | Azadirachta indica | 27 |
| Vachellia nilotica | Azadirachta indica | 26 |
| Azadirachta indica | Vachellia nilotica | 26 |
| Vachellia nilotica | Prosopis cineraria | 18 |
| Prosopis cineraria | Azadirachta indica | 18 |
| Azadirachta indica | Ailanthus excelsa | 16 |
| Dalbergia sissoo | Azadirachta indica | 14 |
| Azadirachta indica | Prosopis cineraria | 13 |
| Mangifera indica | Azadirachta indica | 8 |