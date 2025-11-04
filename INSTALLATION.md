# Installation Guide for find_zones.py

## For Linux VM / Conda Environment

### Step 1: Install System Dependencies (libspatialindex)

```bash
# Install libspatialindex using conda (RECOMMENDED)
conda install -c conda-forge libspatialindex
```

### Step 2: Install Core Geospatial Packages via Conda

```bash
# Install geospatial packages with conda (better compatibility)
conda install -c conda-forge geopandas rasterio shapely rtree
```

### Step 3: Install PyTorch and TorchGeo

```bash
# Install PyTorch (CPU version for VM without GPU)
conda install pytorch torchvision cpuonly -c pytorch

# Or for GPU version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TorchGeo
pip install torchgeo
```

### Step 4: Install Computer Vision Packages

```bash
# Install OpenCV
conda install -c conda-forge opencv

# Install Pillow and imagehash
conda install pillow
pip install imagehash
```

### Step 5: Install Remaining Dependencies

```bash
# Install other packages
conda install pandas matplotlib numpy
```

## Complete One-Line Installation (Recommended)

```bash
# Install everything at once with conda
conda install -c conda-forge libspatialindex geopandas rasterio shapely rtree opencv pillow pandas matplotlib numpy && \
conda install pytorch torchvision cpuonly -c pytorch && \
pip install torchgeo imagehash
```

## Verify Installation

```bash
# Test all imports
python -c "
import cv2
import geopandas as gpd
import imagehash
import torch
from torchgeo.datasets import RasterDataset
from shapely import Point
import rasterio
print('All packages imported successfully!')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'GeoPandas: {gpd.__version__}')
"
```

## Troubleshooting

### Issue: "Could not load libspatialindex_c library"
```bash
conda install -c conda-forge libspatialindex
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"
```bash
conda install -c conda-forge opencv
# NOT: pip install cv2 (this won't work)
```

### Issue: "ModuleNotFoundError: No module named 'torchgeo'"
```bash
pip install torchgeo
```

### Issue: GDAL/Rasterio errors
```bash
# Reinstall with conda-forge
conda install -c conda-forge rasterio gdal
```

### Issue: Pandas compilation errors (old GCC)
```bash
# Use conda instead of pip
conda install -c conda-forge pandas
```

## Minimal Installation (Only for find_zones.py)

If you only need to run `find_zones.py`, install just these:

```bash
conda install -c conda-forge libspatialindex geopandas rasterio shapely rtree opencv pillow && \
conda install pytorch torchvision cpuonly -c pytorch && \
pip install torchgeo imagehash
```

## After Installation

Run the zone detection:
```bash
python find_zones.py \
  --state karnataka \
  --source-dir ../tifs_for_labeling/karnataka_top_10_tifs/ \
  --shapefile ./district_shapefiles/2011_Dist.shp \
  --tiles-dir ../new_karnataka_california_cvat/kejri_trees_karnataka/images/default/ \
  --output karnataka_final_zones.csv
```
