#!/usr/bin/env python3
"""
Corrected Karnataka Zone Dictionary

This dictionary uses the exact district names from the shapefile 
'./district_shapefiles/2011_Dist.shp' with state 'Karnataka'.

Copy this dictionary into your find_zones.py to replace KARNATAKA_ZONES.
"""

CORRECTED_KARNATAKA_ZONES = {
    # Exact matches from shapefile
    'Bagalkot': 'NORTHERN DRY',  # Need to determine zone
    'Bangalore': 'EASTERN DRY',  # This is likely Bengaluru Urban
    'Bangalore Rural': 'EASTERN DRY',  # Matches BENGALURU RURAL
    'Belgaum': 'WESTERN TRANSITION',  # Need to determine zone
    'Bellary': 'NORTH EAST DRY',  # Matches BALLARI
    'Bidar': 'NORTHERN DRY',  # Need to determine zone
    'Bijapur': 'NORTHERN DRY',  # Need to determine zone
    'Chamrajnagar': 'SOUTHERN DRY',  # Need to determine zone
    'Chikkaballapura': 'EASTERN DRY',  # Matches CHIKKABALLAPURA
    'Chikmagalur': 'SOUTHERN TRANSITION',  # Need to determine zone
    'Chitradurga': 'CENTRAL DRY',  # Matches CHITRADURGA
    'Dakshina Kannada': 'COASTAL',  # Matches DAKSHINA KANNADA
    'Davanagere': 'CENTRAL DRY',  # Matches DAVANGERE
    'Dharwad': 'WESTERN TRANSITION',  # Matches DHARWAD
    'Gadag': 'NORTHERN DRY',  # Matches GADAG
    'Gulbarga': 'NORTHERN DRY',  # Need to determine zone
    'Hassan': 'SOUTHERN TRANSITION',  # Matches HASSAN
    'Haveri': 'NORTHERN DRY',  # Need to determine zone
    'Kodagu': 'SOUTHERN DRY',  # Matches KODAGU
    'Kolar': 'EASTERN DRY',  # Matches KOLAR
    'Koppal': 'NORTH EAST DRY',  # Matches KOPPAL
    'Mandya': 'SOUTHERN DRY',  # Matches MANDYA
    'Mysore': 'SOUTHERN DRY',  # Need to determine zone
    'Raichur': 'NORTH EAST DRY',  # Matches RAICHUR
    'Ramanagara': 'EASTERN DRY',  # Matches RAMANAGARAM
    'Shimoga': 'SOUTHERN TRANSITION',  # Matches SHIVAMOGGA
    'Tumkur': 'CENTRAL DRY',  # Matches TUMAKURU
    'Udupi': 'COASTAL',  # Matches UDUPI
    'Uttara Kannada': 'HILL',  # Matches UTTARA KANNADA
    'Yadgir': 'NORTH EAST DRY',  # Matches YADGIR
}

# Districts that need zone determination (marked as 'NORTHERN DRY' temporarily)
NEEDS_ZONE_DETERMINATION = [
    'Bagalkot', 'Belgaum', 'Bidar', 'Bijapur', 'Chamrajnagar', 
    'Chikmagalur', 'Gulbarga', 'Haveri', 'Mysore'
]

print("✅ Corrected Karnataka Zone Dictionary Ready!")
print(f"📍 Total districts: {len(CORRECTED_KARNATAKA_ZONES)}")
print(f"⚠️  Districts needing zone determination: {len(NEEDS_ZONE_DETERMINATION)}")
print("\n📋 Districts that need manual zone assignment:")
for district in NEEDS_ZONE_DETERMINATION:
    print(f"  - {district}: {CORRECTED_KARNATAKA_ZONES[district]}")

print(f"\n🎯 Ready to copy into find_zones.py!")
print("Replace the existing KARNATAKA_ZONES dictionary with CORRECTED_KARNATAKA_ZONES")
