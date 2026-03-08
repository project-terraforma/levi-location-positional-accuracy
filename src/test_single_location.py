import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import folium
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Enable caching
ox.settings.use_cache = True
ox.settings.cache_folder = Path("data/osm_cache")
ox.settings.cache_folder.mkdir(exist_ok=True, parents=True)

# ============================================================
# CONFIGURE YOUR TEST LOCATION HERE
# ============================================================
TEST_LOCATION = {
    'place_name': 'La Belle Elaine\'s Bridal',
    'category': 'bridal_shop',
    'address': '1550 Eastlake Ave E',
    'original_lat': 47.6337251,
    'original_lon': -122.3253727,
    'gt_lat': 47.6336835,  # Your manually labeled correct location
    'gt_lon': -122.3254237
}

SEARCH_RADIUS_METERS = 100
MAX_DISTANCE_METERS = 50

# ============================================================
# Import functions from main script
# ============================================================

def download_osm_data_for_point(lat, lon, buffer_m=100):
    """Download buildings, streets, and parking lots"""
    try:
        point = (lat, lon)
        time.sleep(0.5)
        
        buildings = ox.features_from_point(point, tags={'building': True}, dist=buffer_m)
        
        try:
            G = ox.graph_from_point(point, dist=buffer_m, network_type='all')
            streets_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        except:
            streets_gdf = None
        
        try:
            parking_lots = ox.features_from_point(point, tags={'amenity': 'parking'}, dist=buffer_m)
        except:
            parking_lots = None
        
        return buildings, streets_gdf, parking_lots
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None, None


def calculate_street_facing_point(building_geom, street_geom):
    """Find point on building that faces the street"""
    building_boundary = building_geom.exterior
    closest_street_point = street_geom.interpolate(street_geom.project(building_geom.centroid))
    street_facing_point = building_boundary.interpolate(building_boundary.project(closest_street_point))
    return street_facing_point


def calculate_parking_facing_point(building_geom, parking_geom):
    """Find point on building that faces parking lot"""
    building_boundary = building_geom.exterior
    parking_centroid = parking_geom.centroid
    parking_facing_point = building_boundary.interpolate(building_boundary.project(parking_centroid))
    return parking_facing_point


def calculate_repositioning_strategies(building_geom, nearby_streets, parking_geom, original_point):
    """Calculate all repositioning strategies"""
    strategies = {}
    
    centroid = building_geom.centroid
    bounds = building_geom.bounds
    building_boundary = building_geom.exterior
    
    # 1. Centroid
    strategies['Centroid'] = centroid
    
    # 2. Nearest edge
    strategies['Nearest Edge'] = building_boundary.interpolate(building_boundary.project(original_point))
    
    # 3-N. Street-facing points
    if nearby_streets and len(nearby_streets) > 0:
        for i, street_info in enumerate(nearby_streets):
            street_facing_point = calculate_street_facing_point(building_geom, street_info['geometry'])
            if i == 0:
                strategies[f'Street 1 (Closest)'] = street_facing_point
            else:
                strategies[f'Street {i+1}'] = street_facing_point
    
    # Parking-facing
    if parking_geom is not None:
        strategies['Parking Facing'] = calculate_parking_facing_point(building_geom, parking_geom)
    
    # Cardinal edges
    south_point = Point(centroid.x, bounds[1] + (bounds[3] - bounds[1]) * 0.2)
    strategies['South Edge'] = building_boundary.interpolate(building_boundary.project(south_point))
    
    north_point = Point(centroid.x, bounds[3] - (bounds[3] - bounds[1]) * 0.2)
    strategies['North Edge'] = building_boundary.interpolate(building_boundary.project(north_point))
    
    east_point = Point(bounds[2] - (bounds[2] - bounds[0]) * 0.2, centroid.y)
    strategies['East Edge'] = building_boundary.interpolate(building_boundary.project(east_point))
    
    west_point = Point(bounds[0] + (bounds[2] - bounds[0]) * 0.2, centroid.y)
    strategies['West Edge'] = building_boundary.interpolate(building_boundary.project(west_point))
    
    return strategies


def find_best_matching_streets(streets_gdf, streets_proj, building_centroid, address_str=None, top_n=3):
    """Find top N nearest streets, prioritizing address match"""
    if streets_gdf is None or len(streets_gdf) == 0:
        return []
    
    street_distances = streets_proj.distance(building_centroid)
    sorted_indices = street_distances.sort_values().head(top_n * 2).index
    
    nearby_streets = []
    address_matched_street = None
    
    if address_str:
        address_lower = address_str.lower()
        for idx in sorted_indices:
            street_name = streets_gdf.loc[idx].get('name', '')
            if pd.notna(street_name) and street_name.lower() in address_lower:
                address_matched_street = {
                    'idx': idx,
                    'geometry': streets_proj.loc[idx].geometry,
                    'name': street_name,
                    'distance_m': street_distances.loc[idx],
                    'match_method': 'address_match'
                }
                break
    
    if address_matched_street:
        nearby_streets.append(address_matched_street)
        for idx in sorted_indices[:top_n]:
            if idx != address_matched_street['idx']:
                street_name = streets_gdf.loc[idx].get('name', 'Unnamed Street')
                nearby_streets.append({
                    'idx': idx,
                    'geometry': streets_proj.loc[idx].geometry,
                    'name': street_name,
                    'distance_m': street_distances.loc[idx],
                    'match_method': 'distance_based'
                })
                if len(nearby_streets) >= top_n:
                    break
    else:
        for idx in sorted_indices[:top_n]:
            street_name = streets_gdf.loc[idx].get('name', 'Unnamed Street')
            nearby_streets.append({
                'idx': idx,
                'geometry': streets_proj.loc[idx].geometry,
                'name': street_name,
                'distance_m': street_distances.loc[idx],
                'match_method': 'distance_based'
            })
    
    return nearby_streets[:top_n]


# ============================================================
# MAIN PROCESSING
# ============================================================

print("=" * 60)
print("SINGLE LOCATION REPOSITIONING TEST")
print("=" * 60)
print(f"\nTesting: {TEST_LOCATION['place_name']}")
print(f"Category: {TEST_LOCATION['category']}")
print(f"Address: {TEST_LOCATION['address']}")
print(f"Original: ({TEST_LOCATION['original_lat']}, {TEST_LOCATION['original_lon']})")
print(f"Ground Truth: ({TEST_LOCATION['gt_lat']}, {TEST_LOCATION['gt_lon']})")

# Download OSM data
print(f"\nDownloading OSM data (radius: {SEARCH_RADIUS_METERS}m)...")
buildings, streets, parking_lots = download_osm_data_for_point(
    TEST_LOCATION['original_lat'],
    TEST_LOCATION['original_lon'],
    SEARCH_RADIUS_METERS
)

if buildings is None or len(buildings) == 0:
    print("❌ No buildings found!")
    exit(1)

print(f"✓ Found {len(buildings)} buildings")
print(f"✓ Found {len(streets) if streets is not None else 0} streets")
print(f"✓ Found {len(parking_lots) if parking_lots is not None else 0} parking lots")

# Project to metric CRS
original_geom = Point(TEST_LOCATION['original_lon'], TEST_LOCATION['original_lat'])
original_proj = gpd.GeoSeries([original_geom], crs="EPSG:4326").to_crs("EPSG:3857")[0]
buildings_proj = buildings.to_crs("EPSG:3857")

# Find nearest building
distances = buildings_proj.distance(original_proj)
nearest_idx = distances.idxmin()
nearest_distance = distances.min()
building_geom = buildings_proj.loc[nearest_idx].geometry

print(f"\nNearest building: {nearest_distance:.1f}m away")

# Find nearby streets
nearby_streets = []
if streets is not None and len(streets) > 0:
    streets_proj = streets.to_crs("EPSG:3857")
    nearby_streets = find_best_matching_streets(
        streets, streets_proj, building_geom.centroid,
        TEST_LOCATION['address'], top_n=3
    )
    print(f"\nTop {len(nearby_streets)} streets:")
    for i, s in enumerate(nearby_streets):
        print(f"  {i+1}. {s['name']} - {s['distance_m']:.1f}m ({s['match_method']})")

# Find nearest parking
nearest_parking = None
if parking_lots is not None and len(parking_lots) > 0:
    parking_proj = parking_lots.to_crs("EPSG:3857")
    parking_distances = parking_proj.distance(building_geom.centroid)
    if len(parking_distances) > 0:
        nearest_parking_idx = parking_distances.idxmin()
        nearest_parking = parking_proj.loc[nearest_parking_idx]
        print(f"\nNearest parking: {parking_distances.min():.1f}m away")

# Calculate all strategies
print("\nCalculating repositioning strategies...")
strategies = calculate_repositioning_strategies(
    building_geom,
    nearby_streets,
    nearest_parking.geometry if nearest_parking is not None else None,
    original_proj
)

print(f"✓ Generated {len(strategies)} strategies")

# ============================================================
# CREATE VISUALIZATION MAP
# ============================================================

print("\nCreating visualization map...")

# Center map on original location
m = folium.Map(
    location=[TEST_LOCATION['original_lat'], TEST_LOCATION['original_lon']],
    zoom_start=19,
    tiles='CartoDB positron'
)

# Add building
folium.GeoJson(
    buildings.loc[nearest_idx].geometry,
    style_function=lambda x: {
        'fillColor': 'lightblue',
        'color': 'darkblue',
        'weight': 3,
        'fillOpacity': 0.4
    },
    tooltip="Matched Building"
).add_to(m)

# Add streets
street_colors = ['purple', 'darkviolet', 'mediumorchid']
if streets is not None:
    for i, street_info in enumerate(nearby_streets):
        idx = street_info['idx']
        color = street_colors[i] if i < len(street_colors) else 'gray'
        folium.GeoJson(
            streets.loc[idx].geometry,
            style_function=lambda x, c=color: {'color': c, 'weight': 4, 'opacity': 0.8},
            tooltip=f"{street_info['name']} ({street_info['match_method']})"
        ).add_to(m)

# Add parking if exists
if nearest_parking is not None:
    folium.GeoJson(
        parking_lots.loc[nearest_parking_idx].geometry,
        style_function=lambda x: {
            'fillColor': 'orange',
            'color': 'darkorange',
            'weight': 2,
            'fillOpacity': 0.3
        },
        tooltip="Parking Lot"
    ).add_to(m)

# Add original location (RED)
folium.Marker(
    location=[TEST_LOCATION['original_lat'], TEST_LOCATION['original_lon']],
    popup="<b>Original Location</b><br>(Incorrect)",
    icon=folium.Icon(color='red', icon='times'),
    tooltip="Original (Incorrect)"
).add_to(m)

# Add ground truth (BLUE)
folium.Marker(
    location=[TEST_LOCATION['gt_lat'], TEST_LOCATION['gt_lon']],
    popup="<b>Ground Truth</b><br>(Manually Labeled)",
    icon=folium.Icon(color='blue', icon='check'),
    tooltip="Ground Truth (Correct)"
).add_to(m)

# Add all strategy points
strategy_colors = {
    'Centroid': 'blue',
    'Nearest Edge': 'lightblue',
    'Parking Facing': 'orange',
    'South Edge': 'pink',
    'North Edge': 'darkred',
    'East Edge': 'lightgreen',
    'West Edge': 'cadetblue'
}

# Calculate distances to ground truth
gt_point = Point(TEST_LOCATION['gt_lon'], TEST_LOCATION['gt_lat'])
gt_proj = gpd.GeoSeries([gt_point], crs="EPSG:4326").to_crs("EPSG:3857")[0]

print("\nStrategy distances to ground truth:")
best_strategy = None
best_distance = float('inf')

for name, point_proj in strategies.items():
    point_wgs84 = gpd.GeoSeries([point_proj], crs="EPSG:3857").to_crs("EPSG:4326")[0]
    distance_to_gt = point_proj.distance(gt_proj)
    
    # Determine color
    if 'Street' in name:
        street_num = int(name.split()[1]) - 1
        color = street_colors[street_num] if street_num < len(street_colors) else 'purple'
    else:
        color = strategy_colors.get(name, 'gray')
    
    # Add to map
    folium.CircleMarker(
        location=[point_wgs84.y, point_wgs84.x],
        radius=6,
        popup=f"<b>{name}</b><br>Distance to GT: {distance_to_gt:.2f}m",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        tooltip=name
    ).add_to(m)
    
    print(f"  {name}: {distance_to_gt:.2f}m")
    
    if distance_to_gt < best_distance:
        best_distance = distance_to_gt
        best_strategy = name

print(f"\n🏆 Best strategy: {best_strategy} ({best_distance:.2f}m from ground truth)")

# Add legend
legend_html = f'''
<div style="position: fixed; top: 10px; right: 10px; width: 250px; 
     background: white; border: 2px solid #666; z-index: 9999;
     font-family: -apple-system, BlinkMacSystemFont, sans-serif;
     font-size: 12px; padding: 14px; border-radius: 8px;
     box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
     <h3 style="margin:0 0 4px 0; font-size:14px; color:#333;">{TEST_LOCATION['place_name']}</h3>
     <p style="margin:0; font-size:10px; color: #666;">{TEST_LOCATION['address']}</p>
     <hr style="margin:8px 0; border:0; border-top:1px solid #ddd;">
     <p style="margin:2px;"><span style="color:#e53935;">✖</span> Original (Incorrect)</p>
     <p style="margin:2px;"><span style="color:#1e88e5;">✔</span> Ground Truth (Correct)</p>
     <hr style="margin:8px 0; border:0; border-top:1px solid #ddd;">
     <p style="margin:2px 0 4px 0; font-weight:bold; font-size:11px; color:#666;">Strategies:</p>
     <p style="margin:2px; font-size: 11px;"><span style="color:blue;">●</span> Centroid</p>
     <p style="margin:2px; font-size: 11px;"><span style="color:purple;">●</span> Street 1-3</p>
     <p style="margin:2px; font-size: 11px;"><span style="color:orange;">●</span> Parking</p>
     <p style="margin:2px; font-size: 11px;"><span style="color:pink;">●</span> Cardinal Edges</p>
     <hr style="margin:8px 0; border:0; border-top:1px solid #ddd;">
     <p style="margin:2px; font-size:11px;"><b>Best Strategy:</b> {best_strategy}</p>
     <p style="margin:2px; font-size:11px;"><b>Error distance:</b> {best_distance:.2f}m</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
output_file = Path("data/test_single_location_map.html")
m.save(str(output_file))

print(f"\n✓ Map saved to: {output_file}")
print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
