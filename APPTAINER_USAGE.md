# Apptainer Container Usage Guide

## 🏗️ Building the Container

### On VM:
```bash
cd /scratch/groups/dlobell/aadityan/SiddLabMisc

# Pull latest code
git pull origin main

# Remove old container
rm -f zones.sif

# Build container (30-60 min)
apptainer build zones.sif start.def
```

---

## 🚀 Running Scripts

The container supports two scripts:
1. **`find_zones`** - Classify tiles into agroclimatic zones
2. **`zone_analysis`** - Analyze zone distribution from CSV files

### General Syntax:
```bash
apptainer run [--nv] zones.sif <script_name> [args...]
```

**Note:** Use `--nv` flag if you have GPU access

---

## 📋 Script 1: find_zones

### Purpose:
Classify 400x400 TIF tiles into agroclimatic zones using hash matching with georeferenced source files.

### Syntax:
```bash
apptainer run [--nv] zones.sif find_zones \
  --state <state> \
  --source-dir <source_dir> \
  --shapefile <shapefile> \
  --tiles-dir <tiles_dir> \
  --output <output_csv>
```

### Required Arguments:
- `--state` - State to process (karnataka or rajasthan)
- `--source-dir` - Directory containing georeferenced source GeoTIFF files
- `--shapefile` - Path to shapefile with district boundaries
- `--tiles-dir` - Directory containing 400x400 TIF tiles to classify
- `--output` - Output CSV file path

### Optional Arguments:
- `--district-key` - Column name in shapefile for districts (default: District)
- `--state-key` - Column name in shapefile for state filtering (default: STATE)
- `--patch-size` - Size of patches to sample (default: 400)
- `--stride` - Stride for patch sampling (default: 400)
- `--black-threshold` - Max ratio of black pixels (default: 0.2)
- `--max-hash-distance` - Max Hamming distance for fuzzy matching (default: 12)

### Examples:

#### Karnataka (CPU):
```bash
apptainer run zones.sif find_zones \
  --state karnataka \
  --source-dir ../tifs_for_labeling/karnataka_top_10_tifs/ \
  --shapefile ./district_shapefiles/2011_Dist.shp \
  --tiles-dir ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  --output karnataka_final_zones.csv
```

#### Karnataka (GPU):
```bash
apptainer run --nv zones.sif find_zones \
  --state karnataka \
  --source-dir ../tifs_for_labeling/karnataka_top_10_tifs/ \
  --shapefile ./district_shapefiles/2011_Dist.shp \
  --tiles-dir ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  --output karnataka_final_zones.csv
```

#### Rajasthan:
```bash
apptainer run --nv zones.sif find_zones \
  --state rajasthan \
  --source-dir ../rajasthan_source_geotiffs/ \
  --shapefile ./district_shapefiles/2011_Dist.shp \
  --tiles-dir ../rajasthan_400x400_tiles/ \
  --output rajasthan_final_zones.csv
```

---

## 📊 Script 2: zone_analysis

### Purpose:
Analyze zone distribution from CSV files and generate bar charts.

### Syntax:
```bash
apptainer run zones.sif zone_analysis \
  <image_dir> \
  <california_csv> \
  <karnataka_csv> \
  <extra_karnataka_csv>
```

### Required Arguments:
1. `image_dir` - Directory containing TIF images
2. `california_csv` - CSV file with California zone mappings
3. `karnataka_csv` - CSV file with Karnataka zone mappings
4. `extra_karnataka_csv` - Additional CSV file with Karnataka zone mappings

### Examples:

#### Analyze Karnataka zones:
```bash
apptainer run zones.sif zone_analysis \
  ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  /dev/null \
  karnataka_final_zones.csv \
  /dev/null
```

#### Analyze mixed Karnataka + California:
```bash
apptainer run zones.sif zone_analysis \
  ../mixed_images/ \
  california_zones.csv \
  karnataka_zones.csv \
  extra_karnataka_zones.csv
```

### Output:
- **Console**: Statistics about zone distribution
- **File**: `zone_distribution.png` - Bar chart visualization

---

## 🎯 Complete Workflow Example

### Step 1: Request Resources
```bash
# With GPU
salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=3:00:00

# Without GPU
salloc --cpus-per-task=8 --mem=32G --time=3:00:00
```

### Step 2: Navigate to Project
```bash
cd /scratch/groups/dlobell/aadityan/SiddLabMisc
```

### Step 3: Run find_zones
```bash
apptainer run --nv zones.sif find_zones \
  --state karnataka \
  --source-dir ../tifs_for_labeling/karnataka_top_10_tifs/ \
  --shapefile ./district_shapefiles/2011_Dist.shp \
  --tiles-dir ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  --output karnataka_final_zones.csv
```

### Step 4: Analyze Results
```bash
apptainer run zones.sif zone_analysis \
  ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  /dev/null \
  karnataka_final_zones.csv \
  /dev/null
```

### Step 5: Download Results
```bash
# On your local machine
scp username@sherlock.stanford.edu:/scratch/groups/dlobell/aadityan/SiddLabMisc/karnataka_final_zones.csv .
scp username@sherlock.stanford.edu:/scratch/groups/dlobell/aadityan/SiddLabMisc/zone_distribution.png .
```

---

## 🔧 Troubleshooting

### Container won't run:
```bash
# Check if container exists
ls -lh zones.sif

# Rebuild if needed
apptainer build zones.sif start.def
```

### GPU not detected:
```bash
# Make sure to use --nv flag
apptainer run --nv zones.sif find_zones ...

# Check GPU availability
nvidia-smi
```

### Out of memory:
```bash
# Request more memory
salloc --mem=64G --time=3:00:00

# Or reduce batch size in script
```

### Files not found:
```bash
# Use absolute paths or check current directory
pwd
ls -la

# Or use relative paths from project root
cd /scratch/groups/dlobell/aadityan/SiddLabMisc
```

---

## 📝 Quick Reference

| **Command** | **Purpose** |
|-------------|-------------|
| `apptainer build zones.sif start.def` | Build container |
| `apptainer run zones.sif` | Show help |
| `apptainer run zones.sif find_zones ...` | Classify tiles |
| `apptainer run zones.sif zone_analysis ...` | Analyze zones |
| `apptainer run --nv zones.sif ...` | Run with GPU |

---

## 🎉 Summary

✅ **Single container** for both scripts  
✅ **Flexible arguments** - no hardcoded paths  
✅ **GPU support** with `--nv` flag  
✅ **Easy to use** - just specify script name and args  
✅ **Reproducible** - same environment every time
