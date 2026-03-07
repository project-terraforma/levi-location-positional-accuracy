# Spatial Repositioning System (Project D)

Author: Levi Laden

# Description

This project explores how to improve the accuracy of place locations by prototyping a spatial repositioning system. It examines different ways a place’s pin location can be defined and works to correct spatial offsets in existing data.

# Goal

Prototype a system that can reposition place locations more accurately.

# Data

~5,000 sample places from the given Overture dataset
Ground truth dataset of 500 places
Open Street Map data from Overpass API
Overture Maps data from S3

# Method

Assess different definitions of place pin locations
Measure spatial offset using ground truth data
Conflate OSM and Overture data to establish geographic context and anchor points
Create a ML model trained on the ground truth data to predict the optimal change (Δlat, Δlon) for correcting pin locations
Create a pipeline that uses the ML model to correct pin locations
Finalize pipeline with post-processing allignment for clustered places

# Model Results

Achieved a mean error of 8.4 meters (77.3% improvement over baseline of 37.1 meters)
Beat nearest edge strategy by 0.67 meters
Cross-validated Direct Regressor:
Mean distance to GT: 8.41m
Median distance to GT: 4.36m
Trimmed mean (drop top 5%): 6.09m
Std across folds: 0.87m

## Overview of the Pipeline

1. **Download Data**: Fetch raw place data (and optionally buildings/addresses) from Overture Maps.
2. **Context Conflation**: Fetch matching buildings, addresses, streets, and parking lots from OSM and Overture. Features are conflated to establish the surrounding physical environment of each place.
3. **Feature Generation**: Compute multiple logical "anchor" points for a place (e.g., building centroid, street-facing edge, parking-facing edge, cardinal edges).
4. **ML Prediction**: A trained Gradient Boosting Regressor takes these features and predicts the optimal coordinate shift (Δlat, Δlon) to place the pin accurately.
5. **Alignment**: Post-process the ML predictions by alligning points along nearby streets or parking lots to improve uniformity.

## Project Structure

```
levi-location-positional-accuracy/
├── src/
│   ├── pipeline.py                 # Main orchestrator script for the end-to-end process
│   │
│   ├── models/                     # Machine Learning components
│   │   ├── train_model.py          # Trains the direct regressor model
│   │   ├── predict_location.py     # Predicts on single or batch locations
│   │
│   └── utils/                      # Data processing and geospatial utilities
│       ├── download_overture.py    # Fetches Overture Maps data
│       ├── get_conflated_repositioning.py # Merges OSM/Overture contexts & computes strategies
│       ├── get_osm_repositioning.py# OSM-specific data fetching
│       ├── align_places.py         # Post-prediction facade alignment logic
│       ├── get_data_offset.py      # Computes distances between datasets
│       ├── prepare_data.py         # Data preparation utilities
│       └── visualize_predictions.py# Map generation tools
│       └── test_single_location.py # Test script with Folium map visualization
│
├── data/
│   ├── models/                     # Saved joblib models and model info JSONs
│   ├── pipeline_results/           # Output CSVs and HTML maps from the pipeline
│   ├── raw_overture/               # Downloaded bounding-box data
│   ├── osm_cache/                  # osmnx cache to speed up repeated queries
│   └── overture_cache/             # Local parquet cache for Overture chunks
```

## Usage

### 1. The Complete Pipeline

To run the entire end-to-end process for a specific bounding box (Download -> Conflate -> Predict -> Align), use the `pipeline.py` script:

```bash
python src/pipeline.py \
    --min_lon -122.029696 \
    --min_lat 36.972250 \
    --max_lon -122.023344 \
    --max_lat 36.976852
```

Outputs will be saved in `data/pipeline_results/` including a CSV of the results and an interactive `.html` map.

### 2. Training the Model

If you have ground truth data and need to retrain the ML model:

```bash
python src/models/train_model.py
```

This script will read from the conflated features CSV, calculate oracle baselines, train the regressor, and save the updated `.joblib` model to `data/models/`.
