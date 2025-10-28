#!/usr/bin/env python3
"""
Compare classification results with ground truth labels
Provides detailed analysis of model performance
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from collections import Counter


def analyze_results(results_csv: str):
    """
    Analyze classification results and print detailed report
    
    Args:
        results_csv: Path to classification results CSV file
    """
    # Load results
    df = pd.read_csv(results_csv)
    
    print("=" * 80)
    print("TREE SPECIES CLASSIFICATION ANALYSIS")
    print("=" * 80)
    print(f"Results file: {results_csv}")
    print(f"Total images: {len(df)}")
    print()
    
    # Overall statistics
    successful = df[df['success'] == True]
    failed = df[df['success'] == False]
    
    print("=" * 80)
    print("PROCESSING STATISTICS")
    print("=" * 80)
    print(f"Successfully classified: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
    print(f"Failed classifications: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")
    
    if len(failed) > 0:
        print("\nFailure reasons:")
        error_counts = failed['error'].value_counts()
        for error, count in error_counts.items():
            print(f"  - {error}: {count}")
    
    print()
    
    if len(successful) == 0:
        print("No successful classifications to analyze.")
        return
    
    # Token usage
    total_tokens = successful['tokens_used'].sum()
    avg_tokens = successful['tokens_used'].mean()
    
    print("=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average per image: {avg_tokens:.0f}")
    print(f"Estimated cost (GPT-4o):")
    print(f"  Input tokens (~70%): ${(total_tokens * 0.7 * 2.5 / 1000000):.3f}")
    print(f"  Output tokens (~30%): ${(total_tokens * 0.3 * 10 / 1000000):.3f}")
    print(f"  Total estimated: ${(total_tokens * 0.7 * 2.5 + total_tokens * 0.3 * 10) / 1000000:.3f}")
    print()
    
    # Confidence distribution
    print("=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)
    confidence_counts = successful['confidence'].value_counts()
    for conf, count in confidence_counts.items():
        print(f"{conf.capitalize()}: {count} ({count/len(successful)*100:.1f}%)")
    print()
    
    # Accuracy analysis
    print("=" * 80)
    print("ACCURACY ANALYSIS")
    print("=" * 80)
    
    # Exact matches (case-insensitive)
    successful['pred_lower'] = successful['predicted_scientific_name'].str.lower().str.strip()
    successful['true_lower'] = successful['true_label'].str.lower().str.strip()
    exact_matches = successful[successful['pred_lower'] == successful['true_lower']]
    
    accuracy = len(exact_matches) / len(successful) * 100
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Correct: {len(exact_matches)}/{len(successful)}")
    print()
    
    # Per-species accuracy
    print("Per-species performance:")
    print("-" * 80)
    species_groups = successful.groupby('true_label')
    
    species_stats = []
    for species, group in species_groups:
        total = len(group)
        correct = len(group[group['pred_lower'] == group['true_lower']])
        acc = correct / total * 100
        species_stats.append({
            'species': species,
            'total': total,
            'correct': correct,
            'accuracy': acc
        })
    
    # Sort by total count
    species_stats.sort(key=lambda x: x['total'], reverse=True)
    
    for stat in species_stats:
        print(f"{stat['species']:<30} | "
              f"Correct: {stat['correct']:>2}/{stat['total']:<2} | "
              f"Accuracy: {stat['accuracy']:>5.1f}%")
    print()
    
    # Confusion analysis
    print("=" * 80)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    misclassified = successful[successful['pred_lower'] != successful['true_lower']]
    
    if len(misclassified) > 0:
        print(f"Total misclassifications: {len(misclassified)}")
        print("\nMost common misclassifications:")
        print("-" * 80)
        
        confusion_pairs = misclassified.groupby(['true_label', 'predicted_scientific_name']).size()
        confusion_pairs = confusion_pairs.sort_values(ascending=False).head(10)
        
        for (true_label, pred_label), count in confusion_pairs.items():
            print(f"{true_label:<30} → {pred_label:<30} | {count} times")
        
        print("\nMisclassified examples:")
        print("-" * 80)
        for idx, row in misclassified.head(5).iterrows():
            print(f"File: {row['filepath']}")
            print(f"  True: {row['true_label']}")
            print(f"  Predicted: {row['predicted_scientific_name']} (confidence: {row['confidence']})")
            if pd.notna(row['visible_features']):
                features = json.loads(row['visible_features'])
                print(f"  Features seen: {', '.join(features[:3])}")
            print()
    else:
        print("No misclassifications! Perfect accuracy!")
    
    print()
    
    # Confidence vs. Accuracy
    print("=" * 80)
    print("CONFIDENCE vs. ACCURACY")
    print("=" * 80)
    
    for conf in ['high', 'medium', 'low']:
        conf_subset = successful[successful['confidence'] == conf]
        if len(conf_subset) > 0:
            conf_correct = len(conf_subset[conf_subset['pred_lower'] == conf_subset['true_lower']])
            conf_acc = conf_correct / len(conf_subset) * 100
            print(f"{conf.capitalize()} confidence: {conf_correct}/{len(conf_subset)} correct ({conf_acc:.1f}%)")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tree species classification results"
    )
    parser.add_argument(
        "results_csv",
        help="Path to classification results CSV file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_csv).exists():
        print(f"Error: File not found: {args.results_csv}")
        return
    
    analyze_results(args.results_csv)


if __name__ == "__main__":
    main()

