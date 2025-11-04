# Apptainer Container Build Guide

## Key Changes for Memory Efficiency

### 1. **Mamba Solver**
- Uses `conda-libmamba-solver` for faster, more memory-efficient dependency resolution
- Significantly reduces memory usage during package installation

### 2. **`--no-update-deps` Flag**
- Critical flag that prevents conda from re-solving all dependencies
- Each package is installed without updating existing dependencies
- Dramatically reduces memory usage during installation

### 3. **Installation Order**
Packages are installed in dependency order:
1. `libspatialindex` (C library for spatial indexing)
2. `pandas` (data processing)
3. `matplotlib` (plotting)
4. `shapely` (geometry operations)
5. `rtree` (spatial indexing, depends on libspatialindex)
6. `gdal` (geospatial data abstraction library)
7. `rasterio` (raster I/O, depends on GDAL)
8. `fiona` (vector I/O, depends on GDAL)
9. `pyproj` (coordinate transformations)
10. `geopandas` (combines all above)
11. `pytorch`, `torchvision` (deep learning)
12. `torchgeo`, `imagehash`, `opencv-python-headless` (via pip)

### 4. **Pip Packages**
- Installed via pip with `--no-cache-dir` to save space
- Pure Python packages that don't need compilation

## Build Instructions

### Step 1: Request Adequate Resources
```bash
# Request a node with sufficient memory and time
salloc --mem=32G --time=3:00:00

# Wait for allocation
# Once allocated, you'll see: "salloc: Nodes sh02-01n57 are ready for job"
```

### Step 2: Navigate to Project Directory
```bash
cd /scratch/groups/dlobell/aadityan/SiddLabMisc
```

### Step 3: Ensure Latest Code
```bash
git pull origin main
```

### Step 4: Build Container
```bash
# Build the container (will take 30-60 minutes)
apptainer build zones.sif start.def
```

### Step 5: Verify Build
```bash
# Check container size
ls -lh zones.sif

# View help
apptainer run-help zones.sif

# Test imports interactively
apptainer shell zones.sif
python -c "import cv2, geopandas, torchgeo, imagehash; print('All imports successful!')"
exit
```

## Running find_zones.py

### Karnataka Example
```bash
apptainer run zones.sif \
    --state karnataka \
    --source-dir ../tifs_for_labeling/karnataka_top_10_tifs/ \
    --shapefile ./district_shapefiles/2011_Dist.shp \
    --tiles-dir ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
    --output karnataka_zones.csv
```

### Rajasthan Example
```bash
apptainer run zones.sif \
    --state rajasthan \
    --source-dir ../tifs_for_labeling/rajasthan_source/ \
    --shapefile ./district_shapefiles/2011_Dist.shp \
    --tiles-dir ../rajasthan_tiles/ \
    --output rajasthan_zones.csv
```

### With Custom Parameters
```bash
apptainer run zones.sif \
    --state karnataka \
    --source-dir ./source \
    --shapefile ./districts.shp \
    --tiles-dir ./tiles \
    --output zones.csv \
    --patch-size 400 \
    --stride 400 \
    --max-hash-distance 15
```

## Troubleshooting

### Build Still Fails with OOM
If you still get exit status 137 (OOM):

1. **Request more memory:**
   ```bash
   salloc --mem=64G --time=4:00:00
   ```

2. **Try building on a different node:**
   ```bash
   # Exit current allocation
   exit
   # Request new allocation
   salloc --mem=32G --time=3:00:00
   ```

3. **Check available memory:**
   ```bash
   free -h
   ```

### Build Succeeds but Container Doesn't Work

1. **Test imports:**
   ```bash
   apptainer shell zones.sif
   python -c "import sys; print(sys.version)"
   python -c "import cv2; print('OpenCV:', cv2.__version__)"
   python -c "import geopandas; print('GeoPandas:', geopandas.__version__)"
   python -c "import torchgeo; print('TorchGeo: OK')"
   exit
   ```

2. **Check environment:**
   ```bash
   apptainer exec zones.sif conda list
   ```

### Runtime Errors

1. **File not found errors:**
   - Ensure all paths are absolute or relative to where you run the command
   - Use `pwd` to check current directory

2. **Permission errors:**
   - Apptainer mounts your home directory by default
   - Use `--bind` to mount additional directories:
     ```bash
     apptainer run --bind /scratch:/scratch zones.sif --state karnataka ...
     ```

## Expected Build Time

- **Total build time:** 30-60 minutes
- **Container size:** ~3-5 GB
- **Memory usage during build:** 8-16 GB peak

## Build Progress Indicators

You'll see output like:
```
Installing libspatialindex...
Collecting package metadata (repodata.json): done
Solving environment: done
...
Installing pandas...
Collecting package metadata (repodata.json): done
Solving environment: done
...
```

Each package installation takes 2-5 minutes.

## Post-Build

Once built successfully:
1. Container is portable - can copy `zones.sif` to other systems
2. No need to rebuild unless code changes
3. All dependencies are frozen in the container
4. Reproducible across different environments

## Summary

This build strategy:
- ✅ Uses mamba solver for efficiency
- ✅ Installs packages one-by-one
- ✅ Uses `--no-update-deps` to prevent dependency re-solving
- ✅ Installs in correct dependency order
- ✅ Cleans up after installation
- ✅ Should work on VMs with 32GB+ memory
