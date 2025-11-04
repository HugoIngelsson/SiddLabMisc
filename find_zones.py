#!/usr/bin/env python3
"""
Find Zones Script
Maps 400x400 TIF images to agroclimatic zones using hash matching with georeferenced source files.

Handles mixed directories: Images with geolocations outside the shapefile boundaries are automatically
ignored (e.g., California images in a Karnataka+California mixed directory).
"""

import os
import sys
import argparse
import cv2
import geopandas as gpd
import imagehash
import torch
from PIL import Image
from shapely import Point
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler
from rasterio.transform import from_bounds
from multiprocessing import Pool, cpu_count
from functools import partial

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
else:
    print(f"No GPU detected, using CPU")


class GeoTiffDataset(RasterDataset):
    """Custom TorchGeo dataset for loading georeferenced GeoTIFF files."""
    filename_glob = "*.tif*"
    filename_regex = r"()"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = True
    all_bands = ["red", "green", "blue"]
    rgb_bands = ["red", "green", "blue"]
    
    def __getitem__(self, index):
        sample_dict = super().__getitem__(index)
        
        # Compute the affine transform for this image
        image = sample_dict['image']
        transform = self.get_transform(sample_dict["bbox"], image.shape[1], image.shape[2])
        sample_dict["transform"] = transform
        return sample_dict
    
    def get_transform(self, bounds, height, width):
        """Convert a BoundingBox to an affine transform."""
        left, top, right, bottom = bounds.minx, bounds.maxy, bounds.maxx, bounds.miny
        return from_bounds(left, bottom, right, top, width, height)


def get_region(shapefile_data, key, point):
    """Find which polygon in shapefile contains the given point."""
    for row_id, row in shapefile_data.iterrows():
        if row['geometry'].intersects(point):
            return row[key]
    return 'UNKNOWN'


def find_closest_hash(target_hash, hash_dict, max_distance=12):
    """Find the closest matching hash if exact match is not found."""
    min_dist = 1000
    best = None
    
    for hash_str in hash_dict:
        h = imagehash.hex_to_hash(hash_str)
        distance = h - target_hash
        if distance < min_dist:
            min_dist = distance
            best = hash_str
    
    if min_dist > max_distance:
        return None, min_dist
    
    return best, min_dist


def build_hash_to_district_mapping(source_dir, shapefile_path, district_key='District', 
                                   target_crs=32643, state_filter=None, state_key='STATE',
                                   patch_size=400, stride=400, black_threshold=0.2):
    """
    Phase 1: Build hash→district mapping from georeferenced source GeoTIFFs.
    
    Args:
        source_dir: Directory containing georeferenced source GeoTIFF files
        shapefile_path: Path to shapefile with district boundaries
        district_key: Column name in shapefile containing district names
        target_crs: Target CRS for shapefile (default: UTM Zone 43N)
        state_filter: Optional state name to filter shapefile
        state_key: Column name for state filtering
        patch_size: Size of patches to sample (default: 400)
        stride: Stride for sampling (default: 400)
        black_threshold: Maximum ratio of black pixels to include patch (default: 0.2)
    
    Returns:
        Dictionary mapping image hashes to district names
    """
    print("\n" + "="*60)
    print("PHASE 1: Building Hash→District Mapping")
    print("="*60)
    
    # Load source GeoTIFFs
    print(f"\nLoading source GeoTIFFs from: {source_dir}")
    dataset = GeoTiffDataset(source_dir)
    print(f"Dataset loaded: {dataset}")
    print(f"Bounding box: {dataset.bounds}")
    
    # Load shapefile
    print(f"\nLoading shapefile: {shapefile_path}")
    shapefile_data = gpd.read_file(shapefile_path).to_crs(target_crs)
    
    # Debug: Print available columns
    print(f"Available columns in shapefile: {list(shapefile_data.columns)}")
    
    # Auto-detect district column (case-insensitive)
    district_col = None
    for col in shapefile_data.columns:
        if col.lower() == district_key.lower():
            district_col = col
            break
    
    if district_col is None:
        print(f"\nError: Column '{district_key}' not found in shapefile.")
        print(f"Available columns: {list(shapefile_data.columns)}")
        sys.exit(1)
    
    print(f"Using district column: '{district_col}'")
    
    # Filter by state if specified
    if state_filter:
        # Auto-detect state column (case-insensitive)
        state_col = None
        for col in shapefile_data.columns:
            if col.lower() == state_key.lower():
                state_col = col
                break
        
        if state_col:
            shapefile_data = shapefile_data[shapefile_data[state_col] == state_filter]
            print(f"Filtered to state: {state_filter} (using column '{state_col}')")
        else:
            print(f"Warning: State column '{state_key}' not found. Processing all districts.")
    
    print(f"Loaded {len(shapefile_data)} districts")
    
    # Update district_key to use the detected column name
    district_key = district_col
    
    # Sample patches and build hash mapping
    print(f"\nSampling {patch_size}x{patch_size} patches (stride={stride})...")
    torch.manual_seed(3)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples,
                           num_workers=4, pin_memory=True)
    
    hash_to_district = {}
    total_patches = 0
    valid_patches = 0
    skipped_black = 0
    skipped_outside = 0
    
    for i, batch in enumerate(dataloader):
        unbinded = unbind_samples(batch)
        sample = unbinded[0]
        im = sample['image']
        
        total_patches += 1
        
        # Skip patches with too many black pixels
        numpixels = im.shape[0] * im.shape[1] * im.shape[2]
        numzero = (im == 0).sum().item()
        ratio = numzero / numpixels
        
        if ratio >= black_threshold:
            skipped_black += 1
            continue
        
        # Get center point from bounds
        bds = sample['bbox']
        pt = Point((bds.minx + bds.maxx) / 2, (bds.miny + bds.maxy) / 2)
        
        # Find district
        district = get_region(shapefile_data, district_key, pt)
        
        if district == 'UNKNOWN':
            # Point is outside shapefile boundaries (e.g., California image when processing Karnataka)
            skipped_outside += 1
            continue
        
        # Compute hash and store mapping
        image = im[0:3].permute(1, 2, 0).numpy().astype('uint8')
        hash_val = imagehash.dhash(Image.fromarray(image))
        hash_to_district[str(hash_val)] = district
        valid_patches += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} patches, {valid_patches} valid mappings...")
    
    print(f"\nTotal patches sampled: {total_patches}")
    print(f"Valid hash→district mappings: {len(hash_to_district)}")
    print(f"Skipped (too many black pixels): {skipped_black}")
    print(f"Skipped (outside shapefile boundaries): {skipped_outside}")
    print(f"Sample mappings: {dict(list(hash_to_district.items())[:5])}")
    
    return hash_to_district


def process_single_tile(filename, tiles_dir, hash_to_district, district_to_zone, max_hash_distance):
    """
    Process a single tile (used for parallel processing).
    
    Returns:
        Tuple of (filename, zone, match_type) where match_type is 'exact', 'fuzzy', 'unknown', or 'error'
    """
    if not (filename.endswith('.tif') or filename.endswith('.tiff')):
        return None
    
    filepath = os.path.join(tiles_dir, filename)
    
    try:
        # Load image and compute hash
        im = cv2.imread(filepath)[:, :, ::-1]  # BGR to RGB
        hash_val = imagehash.dhash(Image.fromarray(im))
        
        # Try exact match first
        if str(hash_val) in hash_to_district:
            district = hash_to_district[str(hash_val)]
            match_type = 'exact'
        else:
            # Try fuzzy match
            best_hash, distance = find_closest_hash(hash_val, hash_to_district, max_hash_distance)
            if best_hash:
                district = hash_to_district[best_hash]
                match_type = 'fuzzy'
            else:
                district = None
                match_type = 'unknown'
        
        # Map district to zone
        if district:
            if district in district_to_zone:
                zone = district_to_zone[district]
            else:
                zone = 'UNKNOWN'
        else:
            zone = 'UNKNOWN'
        
        return (filename, zone, match_type)
        
    except Exception as e:
        return (filename, 'ERROR', 'error')


def classify_tiles(tiles_dir, hash_to_district, district_to_zone, output_csv, max_hash_distance=5, num_workers=None):
    """
    Phase 2: Classify 400x400 tiles by matching their hashes to the hash→district mapping.
    Uses parallel processing for dramatic speedup.
    
    Args:
        tiles_dir: Directory containing 400x400 TIF tiles to classify
        hash_to_district: Dictionary mapping hash strings to district names
        district_to_zone: Dictionary mapping district names to zone names
        output_csv: Path to output CSV file
        max_hash_distance: Maximum Hamming distance for fuzzy matching
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dictionary with classification statistics
    """
    print(f"\n{'='*60}")
    print("PHASE 2: CLASSIFYING TILES (PARALLEL)")
    print(f"{'='*60}")
    print(f"Tiles directory: {tiles_dir}")
    print(f"Hash database size: {len(hash_to_district)} entries")
    print(f"Max hash distance for fuzzy matching: {max_hash_distance}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPUs
    print(f"Using {num_workers} parallel workers")
    
    # Get all tile filenames
    all_files = [f for f in os.listdir(tiles_dir) 
                 if f.endswith('.tif') or f.endswith('.tiff')]
    total = len(all_files)
    print(f"Found {total} tiles to process")
    
    # Process tiles in parallel
    process_func = partial(process_single_tile, 
                          tiles_dir=tiles_dir,
                          hash_to_district=hash_to_district,
                          district_to_zone=district_to_zone,
                          max_hash_distance=max_hash_distance)
    
    results = []
    exact_matches = 0
    fuzzy_matches = 0
    unknown = 0
    errors = 0
    zone_counts = {}
    
    print("\nProcessing tiles...")
    with Pool(processes=num_workers) as pool:
        # Process in chunks to show progress
        chunk_size = 100
        for i, result in enumerate(pool.imap_unordered(process_func, all_files, chunksize=chunk_size)):
            if result is None:
                continue
            
            filename, zone, match_type = result
            results.append((filename, zone))
            
            # Update statistics
            if match_type == 'exact':
                exact_matches += 1
            elif match_type == 'fuzzy':
                fuzzy_matches += 1
            elif match_type == 'unknown':
                unknown += 1
            elif match_type == 'error':
                errors += 1
            
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{total} tiles...")
    
    # Write results to CSV
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w') as f:
        f.write('filename,zone\n')
        for filename, zone in sorted(results):
            f.write(f'{filename},{zone}\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total tiles processed: {total}")
    print(f"Exact hash matches: {exact_matches}")
    print(f"Fuzzy hash matches: {fuzzy_matches}")
    print(f"Unknown/no match: {unknown}")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"\nZone distribution:")
    for zone in sorted(zone_counts.keys()):
        print(f"  {zone}: {zone_counts[zone]}")
    print(f"\nOutput saved to: {output_csv}")
    
    return {
        'total': total,
        'exact_matches': exact_matches,
        'fuzzy_matches': fuzzy_matches,
        'unknown': unknown,
        'zone_counts': zone_counts
    }


def main():
    parser = argparse.ArgumentParser(
        description='Map 400x400 TIF images to agroclimatic zones using hash matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Karnataka (automatically filters shapefile to KARNATAKA)
  python find_zones.py \\
    --state karnataka \\
    --source-dir ./karnataka_source_geotiffs \\
    --shapefile ./india_districts.shp \\
    --tiles-dir ./karnataka_400x400_tiles \\
    --output karnataka_zones.csv
  
  # Rajasthan (automatically filters shapefile to RAJASTHAN)
  python find_zones.py \\
    --state rajasthan \\
    --source-dir ./rajasthan_source_geotiffs \\
    --shapefile ./india_districts.shp \\
    --tiles-dir ./rajasthan_400x400_tiles \\
    --output rajasthan_zones.csv

Available states:
  - karnataka: Karnataka agroclimatic zones
  - rajasthan: Rajasthan agroclimatic zones
  - Use --custom-mapping to override with custom JSON file
        """
    )
    
    # Required arguments
    parser.add_argument('--source-dir', required=True,
                       help='Directory containing georeferenced source GeoTIFF files')
    parser.add_argument('--shapefile', required=True,
                       help='Path to shapefile with district/region boundaries')
    parser.add_argument('--tiles-dir', required=True,
                       help='Directory containing 400x400 TIF tiles to classify')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    
    # Optional arguments
    parser.add_argument('--district-key', default='District',
                       help='Column name in shapefile for district names (default: District)')
    parser.add_argument('--state', choices=['karnataka', 'rajasthan'],
                       required=True,
                       help='State to process (karnataka or rajasthan)')
    parser.add_argument('--custom-mapping', type=str,
                       help='Path to custom district→zone mapping JSON file (overrides state mapping)')
    parser.add_argument('--target-crs', type=int, default=32643,
                       help='Target CRS EPSG code for shapefile (default: 32643 for UTM 43N)')
    parser.add_argument('--state-key', default='STATE',
                       help='Column name in shapefile for state filtering (default: STATE)')
    parser.add_argument('--no-state-filter', action='store_true',
                       help='Disable automatic state filtering (use all districts in shapefile)')
    parser.add_argument('--patch-size', type=int, default=400,
                       help='Size of patches to sample from source (default: 400)')
    parser.add_argument('--stride', type=int, default=400,
                       help='Stride for patch sampling (default: 400)')
    parser.add_argument('--black-threshold', type=float, default=0.2,
                       help='Max ratio of black pixels in valid patches (default: 0.2)')
    parser.add_argument('--max-hash-distance', type=int, default=12,
                       help='Max Hamming distance for fuzzy hash matching (default: 12)')
    
    args = parser.parse_args()
    
    # Define zone mappings
    KARNATAKA_ZONES = {
        'Shimoga': 'SOUTHERN TRANSITION',
        'Uttara Kannada': 'HILL',
        'Davanagere': 'CENTRAL DRY',
        'Chitradurga': 'CENTRAL DRY',
        'Bellary': 'NORTH EAST DRY',
        'Dharwad': 'WESTERN TRANSITION',
        'Gadag': 'NORTHERN DRY',
        'Koppal': 'NORTH EAST DRY',
        'Raichur': 'NORTH EAST DRY',
        'Yadgir': 'NORTH EAST DRY',
        'Kodagu': 'SOUTHERN DRY',
        'Mandya': 'SOUTHERN DRY',
        'Ramanagara': 'EASTERN DRY',
        'Dakshina Kannada': 'COASTAL',
        'Hassan': 'SOUTHERN TRANSITION',
        'Kolar': 'EASTERN DRY',
        'Bangalore Rural': 'EASTERN DRY',
        'Udupi': 'COASTAL',
        'Tumkur': 'CENTRAL DRY',
        'Chikkaballapura': 'EASTERN DRY'
    }
    
    RAJASTHAN_ZONES = {
        # AZ17 - Arid Western Plain and Hyper Arid Partial irrigated
        'BARMER': 'ARID WESTERN PLAIN AND HYPER ARID PARTIAL IRRIGATED',
        'BIKANER': 'ARID WESTERN PLAIN AND HYPER ARID PARTIAL IRRIGATED',
        'JAISALMER': 'ARID WESTERN PLAIN AND HYPER ARID PARTIAL IRRIGATED',
        'CHURU': 'ARID WESTERN PLAIN AND HYPER ARID PARTIAL IRRIGATED',
        'JODHPUR': 'ARID WESTERN PLAIN AND HYPER ARID PARTIAL IRRIGATED',
        
        # AZ18 - Irrigated North Western Plain
        'SRIGANGANAGAR': 'IRRIGATED NORTH WESTERN PLAIN',
        'HANUMANGARH': 'IRRIGATED NORTH WESTERN PLAIN',
        
        # AZ19 - Transitional plain zone of Island drainage
        'NAGOUR': 'TRANSITIONAL PLAIN ZONE OF ISLAND DRAINAGE',
        'SIKAR': 'TRANSITIONAL PLAIN ZONE OF ISLAND DRAINAGE',
        'JHUNJHUNU': 'TRANSITIONAL PLAIN ZONE OF ISLAND DRAINAGE',
        
        # AZ20 - Transitional plain zone of Luni Basin
        'JALOR': 'TRANSITIONAL PLAIN ZONE OF LUNI BASIN',
        'PALI': 'TRANSITIONAL PLAIN ZONE OF LUNI BASIN',
        'SIROHI': 'TRANSITIONAL PLAIN ZONE OF LUNI BASIN',
        
        # AZ21 - Semi arid eastern plain
        'JAIPUR': 'SEMI ARID EASTERN PLAIN',
        'AJMER': 'SEMI ARID EASTERN PLAIN',
        'DAUSA': 'SEMI ARID EASTERN PLAIN',
        'TONK': 'SEMI ARID EASTERN PLAIN',
        
        # AZ22 - Flood prone eastern plain
        'ALWAR': 'FLOOD PRONE EASTERN PLAIN',
        'BHARATPUR': 'FLOOD PRONE EASTERN PLAIN',
        'DHOLPUR': 'FLOOD PRONE EASTERN PLAIN',
        'KARAULI': 'FLOOD PRONE EASTERN PLAIN',
        'SAWAI MADHOPUR': 'FLOOD PRONE EASTERN PLAIN',
        
        # AZ23 - Sub humid southern plain and alluvial hill
        'BHILWARA': 'SUB HUMID SOUTHERN PLAIN AND ALLUVIAL HILL',
        'UDAIPUR': 'SUB HUMID SOUTHERN PLAIN AND ALLUVIAL HILL',
        'CHHITORGARH': 'SUB HUMID SOUTHERN PLAIN AND ALLUVIAL HILL',
        
        # AZ24 - Southern humid plain
        'DUNGARPUR': 'SOUTHERN HUMID PLAIN',
        'BANSWARA': 'SOUTHERN HUMID PLAIN',
        
        # AZ25 - South eastern humid plain
        'KOTA': 'SOUTH EASTERN HUMID PLAIN',
        'JHALAWAR': 'SOUTH EASTERN HUMID PLAIN',
        'BUNDI': 'SOUTH EASTERN HUMID PLAIN',
        'BARAN': 'SOUTH EASTERN HUMID PLAIN'
    }
    
    # Select zone mapping
    if args.custom_mapping:
        import json
        with open(args.custom_mapping, 'r') as f:
            district_to_zone = json.load(f)
        print(f"Using custom zone mapping from: {args.custom_mapping}")
    elif args.state == 'karnataka':
        district_to_zone = KARNATAKA_ZONES
    elif args.state == 'rajasthan':
        district_to_zone = RAJASTHAN_ZONES
    else:
        print(f"Error: Unknown state '{args.state}'")
        sys.exit(1)
    
    # Auto-set state filter based on selected state
    if args.no_state_filter:
        state_filter = None
    else:
        state_filter = args.state.upper()
    
    print("="*60)
    print("FIND ZONES - Hash-Based Zone Classification")
    print("="*60)
    print(f"State: {args.state.upper()}")
    print(f"Source GeoTIFFs: {args.source_dir}")
    print(f"Shapefile: {args.shapefile}")
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output CSV: {args.output}")
    if state_filter:
        print(f"Filtering shapefile to: {state_filter}")
    
    # Phase 1: Build hash→district mapping
    hash_to_district = build_hash_to_district_mapping(
        source_dir=args.source_dir,
        shapefile_path=args.shapefile,
        district_key=args.district_key,
        target_crs=args.target_crs,
        state_filter=state_filter,
        state_key=args.state_key,
        patch_size=args.patch_size,
        stride=args.stride,
        black_threshold=args.black_threshold
    )
    
    # Phase 2: Classify tiles (automatically uses all available CPUs)
    stats = classify_tiles(
        tiles_dir=args.tiles_dir,
        hash_to_district=hash_to_district,
        district_to_zone=district_to_zone,
        output_csv=args.output,
        max_hash_distance=args.max_hash_distance
    )
    
    print(f"\n{'='*60}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
