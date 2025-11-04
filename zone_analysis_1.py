#!/usr/bin/env python3
"""
Zone Analysis Script
Analyzes zone distribution of TIF images using CSV mapping files.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_csv_mappings(csv_paths):
    """Load all CSV files and create filename->zone mapping."""
    mappings = {}
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} mappings from {os.path.basename(csv_path)}")
        
        # Add to mappings dictionary
        for _, row in df.iterrows():
            filename = row['filename']
            zone = row['zone']
            mappings[filename] = zone
    
    return mappings


def analyze_zones(image_dir, csv_files):
    """Analyze zone distribution from image directory and CSV files."""
    
    # Load all CSV mappings
    print("\n=== Loading CSV Files ===")
    mappings = load_csv_mappings(csv_files)
    print(f"Total mappings loaded: {len(mappings)}")
    
    # Initialize zone counters
    karnataka_zones = defaultdict(int)
    california_zones = defaultdict(int)
    
    # Karnataka zone names
    karnataka_zone_names = [
        'EASTERN DRY', 'CENTRAL DRY', 'NORTHERN DRY', 'SOUTHERN TRANSITION',
        'WESTERN TRANSITION', 'COASTAL', 'HILL', 'NORTH EAST DRY', 'SOUTHERN DRY'
    ]
    
    # Process images
    print(f"\n=== Processing Images from {image_dir} ===")
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    total_images = 0
    matched_images = 0
    unmatched_images = 0
    
    for filename in os.listdir(image_dir):
        # Check if it's a TIF file
        if not (filename.endswith('.tif') or filename.endswith('.tiff')):
            continue
        
        total_images += 1
        
        # Look up zone in mappings
        if filename in mappings:
            zone = mappings[filename]
            matched_images += 1
            
            # Categorize as Karnataka or California zone
            if zone in karnataka_zone_names:
                karnataka_zones[zone] += 1
            else:
                california_zones[zone] += 1
        else:
            unmatched_images += 1
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Total TIF images found: {total_images}")
    print(f"Matched images: {matched_images}")
    print(f"Unmatched images (ignored): {unmatched_images}")
    
    print(f"\n=== Karnataka Zones ===")
    karnataka_total = sum(karnataka_zones.values())
    print(f"Total Karnataka images: {karnataka_total}")
    for zone in sorted(karnataka_zones.keys()):
        count = karnataka_zones[zone]
        print(f"  {zone}: {count}")
    
    print(f"\n=== California Zones ===")
    california_total = sum(california_zones.values())
    print(f"Total California images: {california_total}")
    for zone in sorted(california_zones.keys()):
        count = california_zones[zone]
        print(f"  {zone}: {count}")
    
    # Create visualizations
    create_bar_charts(karnataka_zones, california_zones)
    
    return karnataka_zones, california_zones


def create_bar_charts(karnataka_zones, california_zones):
    """Create bar charts for zone distribution."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Karnataka zones chart
    if karnataka_zones:
        zones = sorted(karnataka_zones.keys())
        counts = [karnataka_zones[z] for z in zones]
        
        ax1.bar(range(len(zones)), counts, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(zones)))
        ax1.set_xticklabels(zones, rotation=45, ha='right')
        ax1.set_xlabel('Zone')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Karnataka Zone Distribution (Total: {sum(counts)})')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            ax1.text(i, count, str(count), ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'No Karnataka data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Karnataka Zone Distribution')
    
    # California zones chart
    if california_zones:
        zones = sorted(california_zones.keys())
        counts = [california_zones[z] for z in zones]
        
        ax2.bar(range(len(zones)), counts, color='coral', alpha=0.8)
        ax2.set_xticks(range(len(zones)))
        ax2.set_xticklabels(zones, rotation=45, ha='right')
        ax2.set_xlabel('Zone')
        ax2.set_ylabel('Count')
        ax2.set_title(f'California Zone Distribution (Total: {sum(counts)})')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            ax2.text(i, count, str(count), ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No California data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('California Zone Distribution')
    
    plt.tight_layout()
    plt.savefig('zone_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n=== Chart saved as 'zone_distribution.png' ===")
    plt.show()


def main():
    """Main function to run the analysis."""
    
    if len(sys.argv) < 3:
        print("Usage: python zone_analysis_1.py <image_dir> <csv_file1> [csv_file2] [csv_file3] ...")
        print("\nExamples:")
        print("  # Single CSV file")
        print("  python zone_analysis_1.py ./images ./karnataka_zones.csv")
        print("")
        print("  # Multiple CSV files")
        print("  python zone_analysis_1.py ./images ./california_zones.csv ./karnataka_zones.csv")
        print("")
        print("  # Three CSV files")
        print("  python zone_analysis_1.py ./images ./california_zones.csv ./karnataka_zones.csv ./extra_zones.csv")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    csv_files = sys.argv[2:]  # All remaining arguments are CSV files
    
    print("=" * 60)
    print("Zone Analysis Script")
    print("=" * 60)
    print(f"Image directory: {image_dir}")
    print(f"CSV files ({len(csv_files)}):")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")
    
    analyze_zones(image_dir, csv_files)


if __name__ == "__main__":
    main()
