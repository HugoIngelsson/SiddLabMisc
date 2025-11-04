#!/usr/bin/env python3
"""
Extract District Names from Shapefile

This script extracts all district names from the shapefile and compares them 
with the hardcoded dictionary in find_zones.py to identify spelling differences.

Usage: python extract_districts.py
"""

import geopandas as gpd
import pandas as pd
import os

def main():
    print("="*60)
    print("EXTRACT DISTRICT NAMES FROM SHAPEFILE")
    print("="*60)
    
    # Load the shapefile
    shapefile_path = './district_shapefiles/2011_Dist.shp'
    
    if not os.path.exists(shapefile_path):
        print(f"❌ Error: Shapefile not found at {shapefile_path}")
        print("Make sure the shapefile exists in the correct location.")
        return
    
    try:
        shapefile_data = gpd.read_file(shapefile_path)
        print(f"✅ Shapefile loaded successfully!")
        print(f"📊 Total districts: {len(shapefile_data)}")
        print(f"📋 Available columns: {list(shapefile_data.columns)}")
    except Exception as e:
        print(f"❌ Error loading shapefile: {e}")
        return
    
    # Get unique district names from the 'DISTRICT' column
    districts = sorted(shapefile_data['DISTRICT'].unique())
    
    print(f"\n📍 Found {len(districts)} unique districts:")
    print("-" * 50)
    
    for i, district in enumerate(districts, 1):
        print(f"{i:3d}. {district}")
    
    # Hardcoded dictionary from find_zones.py
    KARNATAKA_ZONES = {
        'SHIVAMOGGA': 'SOUTHERN TRANSITION',
        'UTTARA  KANNADA': 'HILL',
        'DAVANGERE': 'CENTRAL DRY',
        'CHITRADURGA': 'CENTRAL DRY',
        'BALLARI': 'NORTH EAST DRY',
        'DHARWAD': 'WESTERN TRANSITION',
        'GADAG': 'NORTHERN DRY',
        'KOPPAL': 'NORTH EAST DRY',
        'RAICHUR': 'NORTH EAST DRY',
        'YADGIR': 'NORTH EAST DRY',
        'KODAGU': 'SOUTHERN DRY',
        'MANDYA': 'SOUTHERN DRY',
        'RAMANAGARAM': 'EASTERN DRY',
        'DAKSHINA  KANNADA': 'COASTAL',
        'HASSAN': 'SOUTHERN TRANSITION',
        'KOLAR': 'EASTERN DRY',
        'BENGALURU RURAL': 'EASTERN DRY',
        'UDUPI': 'COASTAL',
        'TUMAKURU': 'CENTRAL DRY',
        'CHIKKABALLAPURA': 'EASTERN DRY'
    }
    
    print(f"\n🔧 Hardcoded Karnataka districts in find_zones.py: {len(KARNATAKA_ZONES)}")
    print("-" * 50)
    for district in sorted(KARNATAKA_ZONES.keys()):
        print(f"  - {district}")
    
    # Find mismatches
    shapefile_districts = set(districts)
    dictionary_districts = set(KARNATAKA_ZONES.keys())
    
    missing_from_dict = shapefile_districts - dictionary_districts
    missing_from_shapefile = dictionary_districts - shapefile_districts
    matches = shapefile_districts & dictionary_districts
    
    print(f"\n📊 COMPARISON RESULTS:")
    print("=" * 50)
    print(f"Districts in shapefile: {len(shapefile_districts)}")
    print(f"Districts in dictionary: {len(dictionary_districts)}")
    print(f"Matching districts: {len(matches)}")
    print(f"Districts missing from dictionary: {len(missing_from_dict)}")
    print(f"Districts missing from shapefile: {len(missing_from_shapefile)}")
    
    # Show mismatches
    if missing_from_dict:
        print(f"\n🔍 Districts in SHAPEFILE but NOT in dictionary:")
        print("-" * 60)
        for district in sorted(missing_from_dict):
            print(f"  - '{district}'")
    
    if missing_from_shapefile:
        print(f"\n⚠️  Districts in DICTIONARY but NOT in shapefile:")
        print("-" * 60)
        for district in sorted(missing_from_shapefile):
            print(f"  - '{district}'")
    
    if matches:
        print(f"\n✅ Districts that MATCH (can be used directly):")
        print("-" * 60)
        for district in sorted(matches):
            print(f"  - '{district}'")
    
    # Filter for Karnataka districts only (case-insensitive)
    karnataka_districts = []
    if 'ST_NM' in shapefile_data.columns:
        # Try different possible spellings for Karnataka
        karnataka_variants = ['KARNATAKA', 'KARNATAKA', 'KARNATAKA ', ' Karnataka']
        karnataka_data = None
        
        for variant in karnataka_variants:
            filtered = shapefile_data[shapefile_data['ST_NM'] == variant]
            if len(filtered) > 0:
                karnataka_data = filtered
                print(f"\n✅ Found Karnataka districts using state name: '{variant}'")
                break
        
        if karnataka_data is not None:
            karnataka_districts = sorted(karnataka_data['DISTRICT'].unique())
        else:
            print(f"\n⚠️  Karnataka not found. Showing all districts instead.")
            karnataka_districts = districts
    else:
        print(f"\n⚠️  No state column found. Showing all districts.")
        karnataka_districts = districts
    
    print(f"\n📍 Districts to compare ({len(karnataka_districts)}):")
    print("-" * 60)
    
    for i, district in enumerate(karnataka_districts, 1):
        # Check if this district exists in our dictionary (case-insensitive)
        district_upper = district.upper()
        found_match = False
        matched_key = None
        
        for dict_key in KARNATAKA_ZONES.keys():
            if dict_key.upper() == district_upper:
                status = "✅ MATCH"
                zone = KARNATAKA_ZONES[dict_key]
                found_match = True
                matched_key = dict_key
                break
        
        if not found_match:
            # Try fuzzy matching for common variations
            district_clean = district.replace(' ', '').replace('.', '').upper()
            for dict_key in KARNATAKA_ZONES.keys():
                dict_clean = dict_key.replace(' ', '').replace('.', '').upper()
                if district_clean == dict_clean:
                    status = "🔄 FUZZY"
                    zone = KARNATAKA_ZONES[dict_key]
                    found_match = True
                    matched_key = f"{dict_key} -> {district}"
                    break
        
        if not found_match:
            status = "❌ MISSING"
            zone = "NEEDS_MAPPING"
            matched_key = None
        
        if status == "✅ MATCH":
            print(f"{i:2d}. {district:25s} | {status:10s} | {zone}")
        elif status == "🔄 FUZZY":
            print(f"{i:2d}. {district:25s} | {status:10s} | {zone} (dict: {matched_key})")
        else:
            print(f"{i:2d}. {district:25s} | {status:10s} | {zone}")
    
    # Export to CSV
    print(f"\n💾 Exporting results to CSV...")
    
    results = []
    for district in karnataka_districts:
        # Check if this district exists in our dictionary (case-insensitive)
        district_upper = district.upper()
        found_match = False
        
        for dict_key in KARNATAKA_ZONES.keys():
            if dict_key.upper() == district_upper:
                status = "MATCH"
                zone = KARNATAKA_ZONES[dict_key]
                found_match = True
                break
        
        if not found_match:
            # Try fuzzy matching for common variations
            district_clean = district.replace(' ', '').replace('.', '').upper()
            for dict_key in KARNATAKA_ZONES.keys():
                dict_clean = dict_key.replace(' ', '').replace('.', '').upper()
                if district_clean == dict_clean:
                    status = "FUZZY"
                    zone = KARNATAKA_ZONES[dict_key]
                    found_match = True
                    break
        
        if not found_match:
            status = "MISSING"
            zone = "NEEDS_MAPPING"
        
        results.append({
            'district': district,
            'status': status,
            'zone': zone
        })
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'karnataka_districts_mapping.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved to: {output_file}")
    print(f"\n📋 Summary:")
    if 'ST_NM' in shapefile_data.columns:
        print(f"  - Total Karnataka districts: {len(karnataka_districts)}")
        print(f"  - Mapped districts: {len(df[df['status'] == 'MATCH'])}")
        print(f"  - Unmapped districts: {len(df[df['status'] == 'MISSING'])}")
    else:
        print(f"  - Total districts: {len(districts)}")
        print(f"  - Mapped districts: {len(df[df['status'] == 'MATCH'])}")
        print(f"  - Unmapped districts: {len(df[df['status'] == 'MISSING'])}")
    
    print(f"\n📄 CSV Preview:")
    print("-" * 40)
    print(df.to_string(index=False))
    
    print(f"\n🎯 Next Steps:")
    print("1. Review the mismatches above")
    print("2. Check the CSV file for detailed comparison")
    print("3. Update the KARNATAKA_ZONES dictionary in find_zones.py")
    print("4. Re-run your zone classification")

if __name__ == "__main__":
    main()
