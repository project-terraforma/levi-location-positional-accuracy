import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
from pathlib import Path
import warnings
import time
import folium
from folium import plugins
warnings.filterwarnings('ignore')

# Enable caching
ox.settings.use_cache = True
ox.settings.cache_folder = Path("data/osm_cache")
ox.settings.cache_folder.mkdir(exist_ok=True, parents=True)
ox.settings.timeout = 180

# Configuration
DATA_DIR = Path("data")
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
OUTPUT_DIR = DATA_DIR / "osm_features"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_DISTANCE_METERS = 50
SEARCH_RADIUS_METERS = 100


def download_osm_data_for_point(lat, lon, buffer_m=100):
    """Download buildings, streets, and parking lots"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                print(f"    ⚠️  Retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                return None, None, None


# ============================================================
# Address Matching Helpers
# ============================================================
def parse_address_parts(address_str):
    """
    Parse an address string into (house_number, street_name).
    Returns (None, None) if parsing fails.
    """
    import re
    if not address_str or not isinstance(address_str, str):
        return None, None
    
    # Clean up
    s = address_str.strip().lower()
    
    # Extract number (starts with digits, maybe suffix like 101a or 101-b)
    # Simple regex for leading number
    match_num = re.match(r'^(\d+[a-z]?(?:-\d+)?)', s)
    house_num = match_num.group(1) if match_num else None
    
    # Extract remainder as street name
    street_name = s
    if house_num:
        street_name = s[len(house_num):].strip()
        # Remove common separators like comma
        street_name = street_name.strip(',- ')
        
    return house_num, street_name


def compute_address_similarity(street1, street2):
    """
    Compute similarity score (0-100) between two street names.
    """
    from difflib import SequenceMatcher
    if not street1 or not street2:
        return 0
    
    # Normalize
    s1 = str(street1).lower().replace('st', 'street').replace('ave', 'avenue').replace('rd', 'road').replace('.', '')
    s2 = str(street2).lower().replace('st', 'street').replace('ave', 'avenue').replace('rd', 'road').replace('.', '')
    
    return SequenceMatcher(None, s1, s2).ratio() * 100


def find_best_matching_building(buildings_gdf, buildings_proj, original_point_proj, address_str=None, max_distance_m=50):
    """
    Find best matching building using prioritized logic:
      1. Exact Match (House Num + Street) - Any distance (within loaded buffer)
      2. Good Address Match (Street) - Within 50m
      3. Nearest Building - Within 50m
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return None, None, 'no_buildings'
    
    distances = buildings_proj.distance(original_point_proj)
    nearest_idx = distances.idxmin()
    nearest_distance = distances.min()
    
    target_num, target_street = parse_address_parts(address_str)
    
    best_exact_match = None         # (idx, dist)
    best_exact_score = -1
    best_fuzzy_match_50m = None     # (idx, dist)
    best_fuzzy_score = -1
    
    # Iterate all buildings to find address matches
    # (Checking all is cheap for typical OSM buffer sizes ~50-100 buildings)
    for idx, row in buildings_gdf.iterrows():
        dist = distances.loc[idx]
        
        # Get building address parts
        b_num = str(row.get('addr:housenumber', '')).lower().strip()
        b_street = str(row.get('addr:street', '')).strip()
        if not b_street:
            # Fallback to name or full address if structured fields missing
            full_addr = str(row.get('addr:full', '')).strip() or str(row.get('name', '')).strip()
            _, b_street_parsed = parse_address_parts(full_addr)
            b_street = b_street_parsed or ''
            
        # 1. Check for Exact Match (Number + Street)
        if target_num and b_num and target_num == b_num:
            street_score = compute_address_similarity(target_street, b_street)
            if street_score > 80:  # High confidence
                # If multiple exact matches, take the closer one or higher score
                if street_score > best_exact_score:
                    best_exact_match = (idx, dist)
                    best_exact_score = street_score
                elif street_score == best_exact_score:
                    if best_exact_match and dist < best_exact_match[1]:
                        best_exact_match = (idx, dist)
        
        # 2. Check for Fuzzy Match within 50m (Street only)
        if dist <= max_distance_m and target_street and b_street:
            street_score = compute_address_similarity(target_street, b_street)
            if street_score > 85:  # Very good street match
                if street_score > best_fuzzy_score:
                    best_fuzzy_match_50m = (idx, dist)
                    best_fuzzy_score = street_score
                elif street_score == best_fuzzy_score:
                    if best_fuzzy_match_50m and dist < best_fuzzy_match_50m[1]:
                        best_fuzzy_match_50m = (idx, dist)

    # --- Selection Logic ---
    
    # 1. Exact Match (Prioritize even if > 50m)
    if best_exact_match:
        return best_exact_match[0], best_exact_match[1], 'address_match_exact'
    
    # 2. Good Fuzzy Match (Only within max_distance)
    if best_fuzzy_match_50m:
        return best_fuzzy_match_50m[0], best_fuzzy_match_50m[1], 'address_match_fuzzy'
    
    # 3. Nearest Building (Only within max_distance)
    if nearest_distance <= max_distance_m:
        return nearest_idx, nearest_distance, 'distance_based'
        
    # 4. Give up
    return nearest_idx, nearest_distance, 'too_far'


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


def calculate_all_repositioning_strategies(building_geom, nearby_streets, parking_geom, original_point):
    """
    Calculate ALL repositioning strategies without selecting one
    Returns dict of strategy_name -> Point geometry (in projected CRS)
    """
    strategies = {}
    
    centroid = building_geom.centroid
    bounds = building_geom.bounds
    building_boundary = building_geom.exterior
    
    # 1. Centroid
    strategies['centroid'] = centroid
    
    # 2. Nearest edge
    strategies['nearest_edge'] = building_boundary.interpolate(building_boundary.project(original_point))
    
    # 3-5. Street-facing points (up to 3 streets)
    if nearby_streets and len(nearby_streets) > 0:
        for i, street_info in enumerate(nearby_streets):
            street_facing_point = calculate_street_facing_point(building_geom, street_info['geometry'])
            strategies[f'street_{i+1}_facing'] = street_facing_point
    
    # 6. Parking-facing
    if parking_geom is not None:
        strategies['parking_facing'] = calculate_parking_facing_point(building_geom, parking_geom)
    
    # 7-10. Cardinal edges
    south_point = Point(centroid.x, bounds[1] + (bounds[3] - bounds[1]) * 0.2)
    strategies['south_edge'] = building_boundary.interpolate(building_boundary.project(south_point))
    
    north_point = Point(centroid.x, bounds[3] - (bounds[3] - bounds[1]) * 0.2)
    strategies['north_edge'] = building_boundary.interpolate(building_boundary.project(north_point))
    
    east_point = Point(bounds[2] - (bounds[2] - bounds[0]) * 0.2, centroid.y)
    strategies['east_edge'] = building_boundary.interpolate(building_boundary.project(east_point))
    
    west_point = Point(bounds[0] + (bounds[2] - bounds[0]) * 0.2, centroid.y)
    strategies['west_edge'] = building_boundary.interpolate(building_boundary.project(west_point))
    
    return strategies


def process_single_location(row):
    """Process a single location and extract all OSM features"""
    # Use ground truth coordinates (matches cache from apply_osm_repositioning.py)
    lat = row['gt_latitude']
    lon = row['gt_longitude']
    
    # Extract original coordinates for reference
    if 'original_geometry_wkt' in row and pd.notna(row['original_geometry_wkt']):
        wkt = row['original_geometry_wkt']
        coords = wkt.replace('POINT (', '').replace(')', '').split()
        original_lon = float(coords[0])
        original_lat = float(coords[1])
    else:
        original_lat = row.get('original_lat', lat)
        original_lon = row.get('original_lon', lon)
    
    place_name = row.get('place_name', 'Unknown')
    
    result = {
        'id': row.get('id'),
        'place_name': place_name,
        'primary_category': row.get('primary_category'),
        'formatted_address': row.get('formatted_address'),
        'confidence': row.get('confidence'),
        'original_lat': original_lat,
        'original_lon': original_lon,
        'gt_latitude': lat,
        'gt_longitude': lon,
    }
    
    try:
        # Download OSM data using ground truth coordinates (leverages cache!)
        buildings, streets, parking_lots = download_osm_data_for_point(lat, lon, SEARCH_RADIUS_METERS)
        
        if buildings is None or len(buildings) == 0:
            result.update({
                'matched_building': False,
                'error': 'No buildings found'
            })
            return result
        
        # Find best matching building
        original_geom = Point(lon, lat)
        original_proj = gpd.GeoSeries([original_geom], crs="EPSG:4326").to_crs("EPSG:3857")[0]
        buildings_proj = buildings.to_crs("EPSG:3857")
        
        address = row.get('formatted_address', None)
        nearest_idx, nearest_distance, building_match_method = find_best_matching_building(
            buildings, buildings_proj, original_proj, address, MAX_DISTANCE_METERS
        )
        
        if building_match_method == 'too_far':
            result.update({
                'matched_building': False,
                'building_distance_m': nearest_distance,
                'error': f'Nearest building {nearest_distance:.1f}m away (> {MAX_DISTANCE_METERS}m)'
            })
            return result
        
        # Get matched building
        building_geom = buildings_proj.loc[nearest_idx].geometry
        building_area = building_geom.area
        
        # Find nearby streets
        nearby_streets = []
        if streets is not None and len(streets) > 0:
            streets_proj = streets.to_crs("EPSG:3857")
            nearby_streets = find_best_matching_streets(
                streets, streets_proj, building_geom.centroid, address, top_n=3
            )
        
        # Find nearest parking
        nearest_parking = None
        if parking_lots is not None and len(parking_lots) > 0:
            parking_proj = parking_lots.to_crs("EPSG:3857")
            parking_distances = parking_proj.distance(building_geom.centroid)
            if len(parking_distances) > 0:
                nearest_parking_idx = parking_distances.idxmin()
                nearest_parking = parking_proj.loc[nearest_parking_idx]
        
        # Calculate ALL strategies
        strategies = calculate_all_repositioning_strategies(
            building_geom,
            nearby_streets,
            nearest_parking.geometry if nearest_parking is not None else None,
            original_proj
        )
        
        # Convert all strategies to WGS84 and add to result
        strategy_data = {}
        for strategy_name, point_proj in strategies.items():
            point_wgs84 = gpd.GeoSeries([point_proj], crs="EPSG:3857").to_crs("EPSG:4326")[0]
            distance_from_gt = original_proj.distance(point_proj)
            
            strategy_data[f'{strategy_name}_lat'] = point_wgs84.y
            strategy_data[f'{strategy_name}_lon'] = point_wgs84.x
            strategy_data[f'{strategy_name}_distance_m'] = distance_from_gt
        
        # Add metadata
        result.update({
            'matched_building': True,
            'building_distance_m': nearest_distance,
            'building_match_method': building_match_method,
            'building_area_sqm': building_area,
            'num_streets': len(nearby_streets),
            'has_parking': nearest_parking is not None,
            'num_strategies': len(strategies),
            'error': None,
            **strategy_data
        })
        
        # Add street metadata
        for i, street_info in enumerate(nearby_streets):
            result[f'street_{i+1}_name'] = street_info['name']
            result[f'street_{i+1}_match_method'] = street_info['match_method']
            result[f'street_{i+1}_distance_m'] = street_info['distance_m']
        
    except Exception as e:
        result.update({
            'matched_building': False,
            'error': str(e)
        })
    
    return result


def create_interactive_map(results_df, output_path):
    """Create interactive map with all strategy points"""
    print("\nCreating interactive map...")
    
    # Filter to successfully matched locations
    matched = results_df[results_df['matched_building'] == True].copy()
    
    if len(matched) == 0:
        print("No matched locations to map!")
        return
    
    # Center map on mean location
    center_lat = matched['gt_latitude'].mean()
    center_lon = matched['gt_longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Strategy colors
    strategy_colors = {
        'centroid': '#3388ff',
        'nearest_edge': '#ff7800',
        'street_1_facing': '#9b59b6',
        'street_2_facing': '#8e44ad',
        'street_3_facing': '#6c3483',
        'parking_facing': '#e67e22',
        'south_edge': '#e74c3c',
        'north_edge': '#c0392b',
        'east_edge': '#27ae60',
        'west_edge': '#16a085'
    }
    
    # Add each location as a feature group
    for idx, row in matched.iterrows():
        # Create feature group for this location
        fg = folium.FeatureGroup(name=f"{row['place_name'][:30]}...")
        
        # Add ground truth marker (green) with click-to-zoom
        marker = folium.CircleMarker(
            location=[row['gt_latitude'], row['gt_longitude']],
            radius=8,
            popup=folium.Popup(f"<b>{row['place_name']}</b><br>{row['formatted_address']}<br><i>Click to zoom</i>", max_width=300),
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.7,
            weight=3
        )
        
        # Add JavaScript to zoom on click
        zoom_js = f"""
        function(e) {{
            var map = e.target._map;
            map.setView([{row['gt_latitude']}, {row['gt_longitude']}], 20);
        }}
        """
        marker.add_child(folium.JavascriptLink("https://code.jquery.com/jquery-3.6.0.min.js"))
        marker.add_to(fg)
        
        # Add custom JavaScript for click event
        click_script = f"""
        <script>
        (function() {{
            var markers = document.querySelectorAll('.leaflet-interactive');
            markers.forEach(function(marker) {{
                marker.addEventListener('click', function(e) {{
                    var lat = {row['gt_latitude']};
                    var lon = {row['gt_longitude']};
                    if (Math.abs(e.latlng.lat - lat) < 0.0001 && Math.abs(e.latlng.lng - lon) < 0.0001) {{
                        e.target._map.setView([lat, lon], 20);
                    }}
                }});
            }});
        }})();
        </script>
        """
        
        # Add all strategy points
        for strategy_name, color in strategy_colors.items():
            lat_col = f'{strategy_name}_lat'
            lon_col = f'{strategy_name}_lon'
            dist_col = f'{strategy_name}_distance_m'
            
            if lat_col in row and pd.notna(row[lat_col]):
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=4,
                    popup=f"<b>{strategy_name.replace('_', ' ').title()}</b><br>Distance: {row[dist_col]:.2f}m",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=2
                ).add_to(fg)
        
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add custom JavaScript for click-to-zoom on all green markers
    zoom_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            var map = null;
            // Find the map object
            for (var key in window) {
                if (window[key] && window[key]._container && window[key]._container.classList.contains('folium-map')) {
                    map = window[key];
                    break;
                }
            }
            
            if (map) {
                map.on('click', function(e) {
                    var clickedElement = e.originalEvent.target;
                    if (clickedElement.tagName === 'path' && clickedElement.getAttribute('stroke') === 'green') {
                        var bounds = clickedElement.getBBox();
                        var svg = clickedElement.ownerSVGElement;
                        var pt = svg.createSVGPoint();
                        pt.x = bounds.x + bounds.width / 2;
                        pt.y = bounds.y + bounds.height / 2;
                        var cursorpt = pt.matrixTransform(svg.getScreenCTM().inverse());
                        var latlng = map.layerPointToLatLng([cursorpt.x, cursorpt.y]);
                        map.setView(latlng, 20, {animate: true, duration: 0.5});
                    }
                });
            }
        }, 1000);
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(zoom_script))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: auto; 
         background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding: 10px">
         <p style="margin:0; font-weight:bold;">Strategy Points</p>
         <hr style="margin:5px 0;">
         <p style="margin:2px;"><span style="color:green;">●</span> Ground Truth (click to zoom)</p>
         <p style="margin:2px;"><span style="color:#3388ff;">●</span> Centroid</p>
         <p style="margin:2px;"><span style="color:#ff7800;">●</span> Nearest Edge</p>
         <p style="margin:2px;"><span style="color:#9b59b6;">●</span> Street 1-3 Facing</p>
         <p style="margin:2px;"><span style="color:#e67e22;">●</span> Parking Facing</p>
         <p style="margin:2px;"><span style="color:#e74c3c;">●</span> Cardinal Edges</p>
         <hr style="margin:5px 0;">
         <p style="margin:2px; font-size:10px;">Click green markers to auto-zoom</p>
         <p style="margin:2px; font-size:10px;">Toggle layers in top-right</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(str(output_path))
    print(f"✓ Map saved to: {output_path}")


def main():
    print("=" * 60)
    print("GET OSM REPOSITIONING DATA (ML Pipeline Step 1)")
    print("=" * 60)
    
    # Load ground truth
    print(f"\nLoading ground truth data from {GROUND_TRUTH_CSV}...")
    gt_df = pd.read_csv(GROUND_TRUTH_CSV, quotechar='"', escapechar='\\')
    
    # Normalize column names to match expected format
    column_mapping = {
        'WKT': 'gt_geometry_wkt',
        'Name': 'place_name',
        'Category': 'primary_category',
        'Address': 'formatted_address',
        'Confidence Score': 'confidence',
        'longitude': 'gt_longitude',
        'latitude': 'gt_latitude'
    }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in gt_df.columns:
            gt_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Extract gt_latitude and gt_longitude from WKT if not present
    if 'gt_latitude' not in gt_df.columns or 'gt_longitude' not in gt_df.columns:
        if 'gt_geometry_wkt' in gt_df.columns:
            def extract_coords(wkt):
                if pd.isna(wkt):
                    return None, None
                coords = wkt.replace('POINT (', '').replace(')', '').split()
                return float(coords[0]), float(coords[1])
            
            gt_df[['gt_longitude', 'gt_latitude']] = gt_df['gt_geometry_wkt'].apply(
                lambda x: pd.Series(extract_coords(x))
            )
    
    print(f"✓ Loaded {len(gt_df)} locations")
    
    print(f"\nProcessing locations...")
    print(f"  Max distance threshold: {MAX_DISTANCE_METERS}m")
    print(f"  Search radius: {SEARCH_RADIUS_METERS}m")
    print(f"  Generating ALL repositioning strategies (no selection)")
    
    # Check for checkpoint
    checkpoint_file = OUTPUT_DIR / "checkpoint_features.csv"
    if checkpoint_file.exists():
        print(f"\n⚠️  Found checkpoint file. Resuming from previous run...")
        existing_results = pd.read_csv(checkpoint_file)
        processed_ids = set(existing_results['id'].values)
        results = existing_results.to_dict('records')
        print(f"  Already processed: {len(results)} locations")
    else:
        results = []
        processed_ids = set()
    
    # Process each location
    for idx, row in gt_df.iterrows():
        if row['id'] in processed_ids:
            continue
            
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(gt_df)} ({idx/len(gt_df)*100:.1f}%)")
        
        result = process_single_location(row)
        results.append(result)
        
        # Save checkpoint every 50 locations
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            print(f"  💾 Checkpoint saved ({len(results)} locations)")
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    matched = results_df['matched_building'].sum()
    total = len(results_df)
    
    print(f"\nTotal locations: {total}")
    print(f"Matched to buildings: {matched} ({matched/total*100:.1f}%)")
    print(f"Not matched: {total - matched} ({(total-matched)/total*100:.1f}%)")
    
    if matched > 0:
        matched_df = results_df[results_df['matched_building']]
        print(f"\nFor matched locations:")
        print(f"  Mean strategies per location: {matched_df['num_strategies'].mean():.1f}")
        print(f"  Mean streets per location: {matched_df['num_streets'].mean():.1f}")
        print(f"  Locations with parking: {matched_df['has_parking'].sum()} ({matched_df['has_parking'].sum()/matched*100:.1f}%)")
        
        # Building matching stats
        print(f"\nBuilding matching methods:")
        addr_match = matched_df[matched_df['building_match_method'] == 'address_match']
        dist_match = matched_df[matched_df['building_match_method'] == 'distance_based']
        print(f"  Address-based: {len(addr_match)} ({len(addr_match)/matched*100:.1f}%)")
        print(f"  Distance-based: {len(dist_match)} ({len(dist_match)/matched*100:.1f}%)")
    
    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    csv_path = OUTPUT_DIR / "osm_repositioning_features.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # Create interactive map
    map_path = OUTPUT_DIR / "repositioning_strategies_map.html"
    create_interactive_map(results_df, map_path)
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nGenerated OSM features for ML pipeline:")
    print(f"  ✓ {matched} locations with full strategy data")
    print(f"  ✓ ~10 repositioning strategies per location")
    print(f"  ✓ Building, street, and parking metadata")
    print(f"  ✓ Interactive map for visualization")
    print("\nNext step: Train ML model to select best strategy")
    print("=" * 60)


if __name__ == "__main__":
    main()
