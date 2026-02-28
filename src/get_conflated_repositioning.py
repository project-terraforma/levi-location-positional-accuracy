import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point
from shapely import wkb
import osmnx as ox
import overturemaps
from pathlib import Path
import warnings
import time
import folium
import os
import ssl
import numpy as np

warnings.filterwarnings('ignore')
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

# Enable caching for OSM
ox.settings.use_cache = True
ox.settings.cache_folder = Path("data/osm_cache")
ox.settings.cache_folder.mkdir(exist_ok=True, parents=True)
ox.settings.timeout = 180

# Configuration
DATA_DIR = Path("data")
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
OUTPUT_DIR = DATA_DIR / "conflated_features"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_DISTANCE_METERS = 100
SEARCH_RADIUS_METERS = 100

# ============================================================
# Fetching & Conflation Logic
# ============================================================

def fetch_osm_context(lat, lon, radius_m):
    """Fetch OSM buildings, addresses, streets, and parking."""
    osm_buildings = gpd.GeoDataFrame(columns=['geometry', 'osm_native_addr'], crs="EPSG:4326")
    osm_addresses = gpd.GeoDataFrame(columns=['geometry', 'osm_point_addr'], crs="EPSG:4326")
    streets_gdf = None
    parking_lots = None
    
    point = (lat, lon)
    
    # Streets
    try:
        G = ox.graph_from_point(point, dist=radius_m, network_type='all')
        streets_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    except:
        pass
        
    # Parking
    try:
        parking_lots = ox.features_from_point(point, tags={'amenity': 'parking'}, dist=radius_m)
    except:
        pass

    # OSM Buildings
    try:
        bldgs = ox.features_from_point(point, tags={'building': True}, dist=radius_m)
        if not bldgs.empty:
            bldgs = bldgs[bldgs.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            osm_buildings = bldgs.to_crs("EPSG:4326")
    except:
        pass
        
    def format_osm_addr(row):
        parts = []
        if 'addr:housenumber' in row and pd.notnull(row['addr:housenumber']):
            parts.append(str(row['addr:housenumber']))
        if 'addr:street' in row and pd.notnull(row['addr:street']):
            parts.append(str(row['addr:street']))
        return " ".join(parts) if parts else None

    if not osm_buildings.empty:
        osm_buildings['osm_native_addr'] = osm_buildings.apply(format_osm_addr, axis=1)
    else:
        osm_buildings['osm_native_addr'] = None

    # OSM Addresses
    try:
        addrs = ox.features_from_point(point, tags={'addr:housenumber': True, 'addr:street': True}, dist=radius_m)
        if not addrs.empty:
            addrs = addrs.to_crs("EPSG:4326").copy()
            # Normalize: use centroid for polygon-tagged addresses so sjoin works correctly
            addrs['geometry'] = addrs.geometry.apply(
                lambda g: g.centroid if g.geom_type != 'Point' else g
            )
            osm_addresses = gpd.GeoDataFrame(addrs, geometry='geometry', crs="EPSG:4326")
            osm_addresses['osm_point_addr'] = osm_addresses.apply(format_osm_addr, axis=1)
    except:
        pass

    return osm_buildings, osm_addresses, streets_gdf, parking_lots

def fetch_overture_data(lat, lon, radius_m):
    """Fetch Overture buildings and addresses with local Parquet caching."""
    lat_buffer = radius_m / 111000
    lon_buffer = radius_m / (111000 * 0.8)
    bbox = (lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)
    
    ov_buildings = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
    ov_addresses = gpd.GeoDataFrame(columns=['geometry', 'ov_addr'], crs="EPSG:4326")
    
    # Setup cache directory
    cache_dir = Path("data/overture_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # We round coordinates to 4 decimals (~11m) so nearby requests hit the same cache file
    cache_key = f"{round(lat, 4)}_{round(lon, 4)}_{radius_m}"
    bldg_cache_file = cache_dir / f"buildings_{cache_key}.parquet"
    addr_cache_file = cache_dir / f"addresses_{cache_key}.parquet"
    
    # --- Fetch Buildings ---
    try:
        if bldg_cache_file.exists():
            ov_buildings = gpd.read_parquet(bldg_cache_file)
        else:
            b_features = overturemaps.record_batch_reader("building", bbox).read_all()
            if b_features.num_rows > 0:
                df = b_features.to_pandas()
                ov_buildings = gpd.GeoDataFrame(
                    df, 
                    geometry=df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, (bytes, bytearray)) else x), 
                    crs="EPSG:4326"
                )
                # Keep only what we need and save to cache
                ov_buildings = ov_buildings[['geometry']]
                ov_buildings.to_parquet(bldg_cache_file)
    except Exception as e:
        pass

    # --- Fetch Addresses ---
    try:
        if addr_cache_file.exists():
            ov_addresses = gpd.read_parquet(addr_cache_file)
        else:
            a_features = overturemaps.record_batch_reader("address", bbox).read_all()
            if a_features.num_rows > 0:
                df = a_features.to_pandas()
                ov_addresses = gpd.GeoDataFrame(
                    df, 
                    geometry=df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, (bytes, bytearray)) else x), 
                    crs="EPSG:4326"
                )
                def format_ov_addr(row):
                    parts = []
                    if 'number' in row and pd.notnull(row['number']):
                        parts.append(str(row['number']))
                    if 'street' in row and pd.notnull(row['street']):
                        parts.append(str(row['street']))
                    return " ".join(parts) if parts else None
                ov_addresses['ov_addr'] = ov_addresses.apply(format_ov_addr, axis=1)
                ov_addresses = ov_addresses.dropna(subset=['ov_addr'])
                # Keep only what we need and save to cache
                ov_addresses = ov_addresses[['geometry', 'ov_addr']]
                ov_addresses.to_parquet(addr_cache_file)
    except Exception as e:
        pass

    return ov_buildings, ov_addresses

def conflate_buildings_and_addresses(osm_bldgs, osm_addrs, ov_bldgs, ov_addrs):
    """Conflate prioritizing Overture Buildings and Addresses."""
    # 1. Buildings (Pri 1: Overture, Pri 2: OSM empty spaces)
    if not ov_bldgs.empty:
        ov_bldgs = ov_bldgs[['geometry']].copy()
        ov_bldgs['building_source'] = 'Overture'
        ov_bldgs['osm_native_addr'] = None
    else:
        ov_bldgs = gpd.GeoDataFrame(columns=['geometry', 'building_source', 'osm_native_addr'], crs="EPSG:4326")
        
    if not osm_bldgs.empty and not ov_bldgs.empty:
        intersecting = gpd.sjoin(osm_bldgs, ov_bldgs[['geometry']], how='inner', predicate='intersects')
        osm_bldgs_unique = osm_bldgs[~osm_bldgs.index.isin(intersecting.index)].copy()
        
        # Transfer native OSM address tags from dropped OSM buildings onto the Overture buildings
        try:
            dropped_osm = osm_bldgs[osm_bldgs.index.isin(intersecting.index)].copy()
            dropped_osm_with_addr = dropped_osm[dropped_osm['osm_native_addr'].notna()]
            if not dropped_osm_with_addr.empty:
                ov_bldgs = ov_bldgs.copy()
                ov_bldgs['osm_native_addr'] = None
                ov_proj = ov_bldgs.to_crs("EPSG:3857")
                osm_proj = dropped_osm_with_addr.to_crs("EPSG:3857")
                xfer = gpd.sjoin(ov_proj[['geometry']], osm_proj[['geometry', 'osm_native_addr']],
                                 how='left', predicate='intersects')
                agg_xfer = xfer.groupby(xfer.index)['osm_native_addr'].apply(
                    lambda x: ', '.join(x.dropna().astype(str)) if not x.dropna().empty else None
                )
                ov_bldgs['osm_native_addr'] = agg_xfer.reindex(ov_bldgs.index).values
        except Exception as e:
            ov_bldgs['osm_native_addr'] = None
    else:
        osm_bldgs_unique = osm_bldgs.copy()
        
    if not osm_bldgs_unique.empty:
        osm_bldgs_unique = osm_bldgs_unique[['geometry', 'osm_native_addr']].copy()
        osm_bldgs_unique['building_source'] = 'OSM'
    else:
        osm_bldgs_unique = gpd.GeoDataFrame(columns=['geometry', 'building_source', 'osm_native_addr'], crs="EPSG:4326")
        
    conflated_bldgs = pd.concat([ov_bldgs, osm_bldgs_unique], ignore_index=True)
    if conflated_bldgs.empty:
        return conflated_bldgs

    conflated_bldgs = gpd.GeoDataFrame(conflated_bldgs, geometry='geometry', crs="EPSG:4326")
    
    # 2. Add Addresses — use a 2m buffer so edge-sitting address points aren't missed
    conflated_proj = conflated_bldgs.to_crs("EPSG:3857")
    conflated_buf = conflated_proj.copy()
    conflated_buf['geometry'] = conflated_buf.geometry.buffer(2)

    if not ov_addrs.empty:
        ov_addrs_proj = ov_addrs.to_crs("EPSG:3857")
        join_ov = gpd.sjoin(conflated_buf, ov_addrs_proj[['geometry', 'ov_addr']], how='left', predicate='contains')
        agg_ov = join_ov.groupby(join_ov.index)['ov_addr'].apply(
            lambda x: ", ".join(x.dropna().unique()) if not x.dropna().empty else None
        )
        conflated_bldgs['joined_ov_addr'] = agg_ov
    else:
        conflated_bldgs['joined_ov_addr'] = None
        
    if not osm_addrs.empty:
        osm_addrs_proj = osm_addrs.to_crs("EPSG:3857")
        join_osm = gpd.sjoin(conflated_buf, osm_addrs_proj[['geometry', 'osm_point_addr']], how='left', predicate='contains')
        agg_osm = join_osm.groupby(join_osm.index)['osm_point_addr'].apply(
            lambda x: ", ".join(x.dropna().unique()) if not x.dropna().empty else None
        )
        conflated_bldgs['joined_osm_addr'] = agg_osm
    else:
        conflated_bldgs['joined_osm_addr'] = None

    # Priority: 1. OSM Native, 2. OSM Address Point, 3. Overture
    def determine_final_addr(row):
        ov_addr = row.get('joined_ov_addr')
        osm_native = row.get('osm_native_addr')
        osm_joined = row.get('joined_osm_addr')
        
        addresses = []
        sources = []
        
        if pd.notnull(osm_native):
            addresses.extend([a.strip() for a in str(osm_native).split(',') if a.strip()])
            sources.append('OSM (Native Building)')
            
        if pd.notnull(osm_joined):
            addresses.extend([a.strip() for a in str(osm_joined).split(',') if a.strip()])
            sources.append('OSM (Address Point)')
            
        if pd.notnull(ov_addr):
            addresses.extend([a.strip() for a in str(ov_addr).split(',') if a.strip()])
            sources.append('Overture')
        
        if not addresses:
            return None, 'None'
        
        return ', '.join(addresses), ' + '.join(sources)

    res = conflated_bldgs.apply(determine_final_addr, axis=1)
    conflated_bldgs['final_address'] = [r[0] for r in res]
    conflated_bldgs['address_source'] = [r[1] for r in res]
    
    conflated_bldgs = conflated_bldgs.dropna(subset=['geometry'])
    return conflated_bldgs

# ============================================================
# Core Strategy Logic
# ============================================================

def parse_address_parts(address_str):
    """Parse an address string into (house_number, street_name)."""
    import re
    if not address_str or not isinstance(address_str, str):
        return None, None
    s = address_str.strip().lower()
    
    # Simple regex for leading number (e.g. "123 main st", "123-a main st")
    match_num = re.match(r'^(\d+[a-z]?(?:-\d+)?)', s)
    house_num = match_num.group(1) if match_num else None
    
    street_name = s
    if house_num:
        street_name = s[len(house_num):].strip()
        street_name = street_name.strip(',- ')
        
    # Remove unit/suite descriptors from the end of the string
    street_name = re.sub(r'\s+(ste|suite|apt|unit|#|rm|room|bldg|building|floor|fl|dept)\b.*$', '', street_name)
    
    return house_num, street_name

def extract_route_number(street_str):
    """Extract a numeric route/highway identifier from a street name."""
    import re
    if not street_str:
        return None
    s = str(street_str).lower().strip()
    # Match patterns like: rt-34, rt 34, route 34, rte 34, hwy 34, highway 34,
    # state hwy 34, us-34, nj-34, sr-34, cr-34, i-34, us route 34, state road 34
    pattern = r'(?:(?:us|nj|ny|ca|tx|fl|pa|oh|il|ga|nc|va|ma|state|county|federal|interstate|i)[\s-]*)?' \
              r'(?:rt|rte|route|rd|road|hwy|highway|sr|cr|blvd|boulevard)[\s.-]*(\d+)'
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    # Also catch bare "NJ-34", "US-1" style state/federal prefixes
    prefix_match = re.search(r'(?:us|nj|ny|ca|tx|fl|pa|oh|il|ga|nc|va|ma|i)-(\d+)', s)
    if prefix_match:
        return prefix_match.group(1)
    return None

def compute_address_similarity(street1, street2):
    """Compute similarity score (0-100) between two street names."""
    from difflib import SequenceMatcher
    if not street1 or not street2:
        return 0
    
    # Route number equivalence check
    r1 = extract_route_number(street1)
    r2 = extract_route_number(street2)
    if r1 and r2 and r1 == r2:
        return 100

    s1 = str(street1).lower().replace('st', 'street').replace('ave', 'avenue').replace('rd', 'road').replace('.', '')
    s2 = str(street2).lower().replace('st', 'street').replace('ave', 'avenue').replace('rd', 'road').replace('.', '')
    return SequenceMatcher(None, s1, s2).ratio() * 100


def find_best_matching_building(buildings_gdf, buildings_proj, original_point_proj, address_str=None, max_distance_m=50):
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return None, None, 'no_buildings'
    
    distances = buildings_proj.distance(original_point_proj)
    nearest_idx = distances.idxmin()
    nearest_distance = distances.min()
    
    target_num, target_street = parse_address_parts(address_str)
    
    exact_candidates = []  # (idx, dist, score) — all buildings passing the threshold
    
    for idx, row in buildings_gdf.iterrows():
        dist = distances.loc[idx]
        
        # Overture conflated address — may be a comma-separated list of multiple addresses
        b_addr_raw = row.get('final_address')
        if not b_addr_raw or pd.isna(b_addr_raw):
            continue
        
        # Split on commas to get each individual address stored on this building
        b_addr_list = [a.strip() for a in str(b_addr_raw).split(',') if a.strip()]
        
        # Test each sub-address for an exact match (house number + street name)
        bldg_best_exact_score = -1
        for b_addr in b_addr_list:
            b_num, b_street = parse_address_parts(b_addr)
            if target_num and b_num and target_num == b_num:
                street_score = compute_address_similarity(target_street, b_street)
                if street_score > 80 and street_score > bldg_best_exact_score:
                    bldg_best_exact_score = street_score
        
        # Collect all buildings that qualify (score > 80) — pick closest later
        if bldg_best_exact_score > 80:
            exact_candidates.append((idx, dist, bldg_best_exact_score))

    # 1. Exact address match — among all qualifying buildings, pick the closest
    if exact_candidates:
        # Sort by distance (ascending), score is just used for qualification not ranking
        exact_candidates.sort(key=lambda x: x[1])
        best_idx, best_dist, _ = exact_candidates[0]
        return best_idx, best_dist, 'address_match_exact'
        
    # 2. Distance-based fallback: snap to closest building within radius
    if nearest_distance <= max_distance_m:
        return nearest_idx, nearest_distance, 'distance_based'

    return nearest_idx, nearest_distance, 'too_far'

def find_best_matching_streets(streets_gdf, streets_proj, building_centroid, address_str=None, top_n=3):
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
                if len(nearby_streets) >= top_n: break
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

def calculate_all_repositioning_strategies(building_geom, nearby_streets, parking_geom, original_point):
    strategies = {}
    centroid = building_geom.centroid
    bounds = building_geom.bounds
    building_boundary = building_geom.exterior
    
    strategies['centroid'] = centroid
    strategies['nearest_edge'] = building_boundary.interpolate(building_boundary.project(original_point))
    
    if nearby_streets and len(nearby_streets) > 0:
        for i, street_info in enumerate(nearby_streets):
            street_geom = street_info['geometry']
            closest_st_pt = street_geom.interpolate(street_geom.project(centroid))
            st_facing_pt = building_boundary.interpolate(building_boundary.project(closest_st_pt))
            strategies[f'street_{i+1}_facing'] = st_facing_pt
            
    if parking_geom is not None:
        p_centroid = parking_geom.centroid
        p_facing_pt = building_boundary.interpolate(building_boundary.project(p_centroid))
        strategies['parking_facing'] = p_facing_pt
    
    s_pt = Point(centroid.x, bounds[1] + (bounds[3] - bounds[1]) * 0.2)
    n_pt = Point(centroid.x, bounds[3] - (bounds[3] - bounds[1]) * 0.2)
    e_pt = Point(bounds[2] - (bounds[2] - bounds[0]) * 0.2, centroid.y)
    w_pt = Point(bounds[0] + (bounds[2] - bounds[0]) * 0.2, centroid.y)
    
    strategies['south_edge'] = building_boundary.interpolate(building_boundary.project(s_pt))
    strategies['north_edge'] = building_boundary.interpolate(building_boundary.project(n_pt))
    strategies['east_edge'] = building_boundary.interpolate(building_boundary.project(e_pt))
    strategies['west_edge'] = building_boundary.interpolate(building_boundary.project(w_pt))
    
    return strategies

def process_single_location(row):
    lat = row['gt_latitude']
    lon = row['gt_longitude']
    
    original_lat = row.get('original_lat', lat)
    original_lon = row.get('original_lon', lon)
    if 'original_geometry_wkt' in row and pd.notna(row['original_geometry_wkt']):
        wkt_str = str(row['original_geometry_wkt'])
        coords = wkt_str.replace('POINT (', '').replace(')', '').split()
        if len(coords) == 2:
            original_lon, original_lat = float(coords[0]), float(coords[1])
    
    result = {
        'id': row.get('id'),
        'place_name': row.get('place_name', 'Unknown'),
        'primary_category': row.get('primary_category'),
        'formatted_address': row.get('formatted_address'),
        'confidence': row.get('confidence'),
        'original_lat': original_lat,
        'original_lon': original_lon,
        'gt_latitude': lat,
        'gt_longitude': lon,
    }
    
    try:
        # 1. Fetch data
        osm_bldgs, osm_addrs, streets, parking_lots = fetch_osm_context(lat, lon, SEARCH_RADIUS_METERS)
        ov_bldgs, ov_addrs = fetch_overture_data(lat, lon, SEARCH_RADIUS_METERS)
        
        # 2. Conflate prioritizing Overture
        conflated_buildings = conflate_buildings_and_addresses(osm_bldgs, osm_addrs, ov_bldgs, ov_addrs)
        
        if conflated_buildings.empty:
            result.update({'matched_building': False, 'error': 'No buildings found'})
            return result
            
        # 3. Best Match
        original_geom = Point(lon, lat)
        original_proj = gpd.GeoSeries([original_geom], crs="EPSG:4326").to_crs("EPSG:3857")[0]
        buildings_proj = conflated_buildings.to_crs("EPSG:3857")
        
        address = row.get('formatted_address', None)
        nearest_idx, nearest_distance, match_method = find_best_matching_building(
            conflated_buildings, buildings_proj, original_proj, address, MAX_DISTANCE_METERS
        )
        
        if match_method == 'too_far':
            result.update({'matched_building': False, 'building_distance_m': nearest_distance, 'error': 'Too far'})
            return result
            
        # Get details
        building_geom = buildings_proj.loc[nearest_idx].geometry
        building_source = conflated_buildings.loc[nearest_idx, 'building_source']
        building_address_source = conflated_buildings.loc[nearest_idx, 'address_source']
        
        # Nearby Streets & Parking
        nearby_streets = []
        if streets is not None and not streets.empty:
            streets_proj = streets.to_crs("EPSG:3857")
            nearby_streets = find_best_matching_streets(streets, streets_proj, building_geom.centroid, address)
            
        nearest_parking = None
        if parking_lots is not None and not parking_lots.empty:
            parking_proj = parking_lots.to_crs("EPSG:3857")
            p_dists = parking_proj.distance(building_geom.centroid)
            if not p_dists.empty:
                nearest_parking = parking_proj.loc[p_dists.idxmin()]
                
        # 4. Strategies
        strategies = calculate_all_repositioning_strategies(
            building_geom, nearby_streets, 
            nearest_parking.geometry if nearest_parking is not None else None, 
            original_proj
        )
        
        strat_data = {}
        for s_name, p_proj in strategies.items():
            p_wgs = gpd.GeoSeries([p_proj], crs="EPSG:3857").to_crs("EPSG:4326")[0]
            strat_data[f'{s_name}_lat'] = p_wgs.y
            strat_data[f'{s_name}_lon'] = p_wgs.x
            strat_data[f'{s_name}_distance_m'] = original_proj.distance(p_proj)
            
        # Convert building geometry back to WGS84 just for the CSV
        building_geom_wgs = gpd.GeoSeries([building_geom], crs="EPSG:3857").to_crs("EPSG:4326")[0]
        
        # Save all surrounding buildings to JSON
        conflated_bldgs_wgs = conflated_buildings.to_crs("EPSG:4326")
        surrounding_buildings = []
        for _, b_row in conflated_bldgs_wgs.iterrows():
            surrounding_buildings.append({
                'wkt': b_row.geometry.wkt,
                'address': b_row.get('final_address', 'No Address'),
                'source': b_row.get('building_source', 'Unknown')
            })
            
        result.update({
            'matched_building': True,
            'building_source': building_source,
            'address_source': building_address_source,
            'building_wkt': building_geom_wgs.wkt,
            'surrounding_buildings': json.dumps(surrounding_buildings),
            'building_distance_m': nearest_distance,
            'building_match_method': match_method,
            'building_area_sqm': building_geom.area,
            'num_streets': len(nearby_streets),
            'has_parking': nearest_parking is not None,
            'num_strategies': len(strategies),
            'error': None,
            **strat_data
        })
        
    except Exception as e:
        result.update({'matched_building': False, 'error': str(e)})
        
    return result

def main():
    print(f"\nLoading ground truth data from {GROUND_TRUTH_CSV}...")
    gt_df = pd.read_csv(GROUND_TRUTH_CSV, quotechar='"', escapechar='\\')
    
    column_mapping = {
        'WKT': 'gt_geometry_wkt', 'Name': 'place_name', 'Category': 'primary_category',
        'Address': 'formatted_address', 'Confidence Score': 'confidence',
        'longitude': 'gt_longitude', 'latitude': 'gt_latitude'
    }
    gt_df.rename(columns=column_mapping, inplace=True)
    
    if 'gt_latitude' not in gt_df.columns:
        def extract_coords(wkt):
            if pd.isna(wkt): return None, None
            coords = wkt.replace('POINT (', '').replace(')', '').split()
            return float(coords[0]), float(coords[1])
        if 'gt_geometry_wkt' in gt_df:
            gt_df[['gt_longitude', 'gt_latitude']] = gt_df['gt_geometry_wkt'].apply(lambda x: pd.Series(extract_coords(x)))
            
    print(f"✓ Loaded {len(gt_df)} locations")
    
    # Check for checkpoint
    checkpoint_file = OUTPUT_DIR / "checkpoint_conflated.csv"
    if checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)
        processed_ids = set(existing['id'].values)
        results = existing.to_dict('records')
        print(f"Resuming {len(results)} locations from checkpoint...")
    else:
        results = []
        processed_ids = set()
        
    for idx, row in gt_df.iterrows():
        if row.get('id') in processed_ids: continue
        
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(gt_df)} ({idx/len(gt_df)*100:.1f}%)")
            
        res = process_single_location(row)
        results.append(res)
        
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            
    results_df = pd.DataFrame(results)
    
    matched = results_df['matched_building'].sum()
    total = len(results_df)
    print(f"\nMatched to buildings: {matched} ({matched/total*100:.1f}%)")
    
    if matched > 0:
        b_src = results_df[results_df['matched_building']]['building_source'].value_counts()
        print(f"\nBuilding Sources utilized:")
        print(b_src)
        
    csv_path = OUTPUT_DIR / "conflated_repositioning_features.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to: {csv_path}")
    
    if checkpoint_file.exists(): checkpoint_file.unlink()

if __name__ == "__main__":
    main()
