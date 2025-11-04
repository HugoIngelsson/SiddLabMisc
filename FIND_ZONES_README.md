# Find Zones - Usage Guide

## Overview
`find_zones.py` implements the exact same logic as `find_tif_zones.ipynb` but as a standalone command-line tool. It maps 400x400 TIF images to agroclimatic zones using hash-based matching with georeferenced source files.

## Installation

```bash
# Install required packages
pip install torch torchvision torchgeo opencv-python Pillow geopandas rasterio shapely imagehash

# Or use conda (recommended for VMs)
conda install -c conda-forge torch torchvision opencv pillow geopandas rasterio shapely
pip install torchgeo imagehash
```

## Basic Usage

```bash
python find_zones.py \
  --source-dir ./source_geotiffs \
  --shapefile ./districts.shp \
  --tiles-dir ./400x400_tiles \
  --output zones_output.csv
```

## Required Arguments

- `--source-dir`: Directory containing georeferenced source GeoTIFF files
- `--shapefile`: Path to shapefile with district/region boundaries (.shp file)
- `--tiles-dir`: Directory containing 400x400 TIF tiles to classify
- `--output`: Output CSV file path

## Optional Arguments

### Shapefile Configuration
- `--district-key`: Column name in shapefile for district names (default: `District`)
- `--target-crs`: Target CRS EPSG code (default: `32643` for UTM Zone 43N)
- `--state-filter`: Filter shapefile to specific state (e.g., `KARNATAKA`)
- `--state-key`: Column name for state filtering (default: `STATE`)

### Zone Mapping
- `--zone-mapping`: Choose predefined mapping (default: `karnataka`)
  - `karnataka`: Karnataka agroclimatic zones
  - `california`: California hardiness zones
  - `custom`: Use custom JSON mapping file
- `--custom-mapping`: Path to custom district→zone mapping JSON file

### Sampling Parameters
- `--patch-size`: Size of patches to sample from source (default: `400`)
- `--stride`: Stride for patch sampling (default: `400`)
- `--black-threshold`: Max ratio of black pixels in valid patches (default: `0.2`)

### Matching Parameters
- `--max-hash-distance`: Max Hamming distance for fuzzy hash matching (default: `12`)

## Examples

### Karnataka Classification
```bash
python find_zones.py \
  --source-dir ./karnataka_source_geotiffs \
  --shapefile ./india_districts.shp \
  --tiles-dir ./karnataka_400x400_tiles \
  --output karnataka_zones.csv \
  --district-key District \
  --state-filter KARNATAKA \
  --zone-mapping karnataka
```

### California Classification
```bash
python find_zones.py \
  --source-dir ./california_source_geotiffs \
  --shapefile ./california_zones.shp \
  --tiles-dir ./california_400x400_tiles \
  --output california_zones.csv \
  --district-key zone \
  --target-crs 4326 \
  --zone-mapping california
```

### Custom Zone Mapping
First create a JSON file `custom_zones.json`:
```json
{
  "DISTRICT_1": "ZONE_A",
  "DISTRICT_2": "ZONE_B",
  "DISTRICT_3": "ZONE_A"
}
```

Then run:
```bash
python find_zones.py \
  --source-dir ./source_geotiffs \
  --shapefile ./districts.shp \
  --tiles-dir ./tiles \
  --output custom_zones.csv \
  --zone-mapping custom \
  --custom-mapping custom_zones.json
```

## Output

The script generates a CSV file with two columns:
```csv
filename,zone
image_001.tif,SOUTHERN TRANSITION
image_002.tif,CENTRAL DRY
image_003.tif,UNKNOWN
```

## How It Works

### Phase 1: Build Hash→District Mapping
1. Loads georeferenced source GeoTIFF files using TorchGeo
2. Samples 400x400 patches from source files
3. For each patch:
   - Gets geographic center coordinates
   - Finds which district contains that point (using shapefile)
   - Computes visual hash (dhash) of the image
   - Stores hash→district mapping

### Phase 2: Classify Tiles
1. Loads your 400x400 tiles (without geographic data)
2. For each tile:
   - Computes visual hash
   - Looks up district in hash dictionary (exact or fuzzy match)
   - Maps district to zone using predefined mapping
   - Writes filename,zone to CSV

## Built-in Zone Mappings

### Karnataka Agroclimatic Zones
```python
{
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
```

## Troubleshooting

### Issue: "No module named 'torchgeo'"
```bash
pip install torchgeo
```

### Issue: "GDAL/Rasterio errors"
```bash
# Use conda for better compatibility
conda install -c conda-forge rasterio gdal
```

### Issue: "Too many UNKNOWN classifications"
- Increase `--max-hash-distance` (try 15-20)
- Check that source GeoTIFFs cover the same geographic area as your tiles
- Verify shapefile CRS matches source GeoTIFF CRS

### Issue: "District names don't match"
- Check district names in your shapefile
- Update the zone mapping dictionary or use custom mapping
- Use `--district-key` to specify correct column name

## Performance Tips

- **Large datasets**: The script processes tiles sequentially. For very large datasets (>10k tiles), consider splitting into batches
- **Memory**: Source GeoTIFF loading is memory-efficient (TorchGeo streams data)
- **Speed**: Hash computation is fast (~100-500 tiles/second on modern hardware)

## Next Steps

After generating the zone CSV files, use `zone_analysis_1.py` to visualize zone distributions:

```bash
python zone_analysis_1.py \
  ./tiles_directory \
  ./california_zones.csv \
  ./karnataka_zones.csv \
  ./extra_karnataka_zones.csv
```
