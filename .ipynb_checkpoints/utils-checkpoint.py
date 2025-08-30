from functools import reduce

import folium
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import seaborn as sns

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.neighbors import KDTree
from xgboost import XGBRegressor


def attach_site_context(
    lat: float,
    lon: float,
    districts_gdf: gpd.GeoDataFrame,
    district_coverage: pd.DataFrame,
    land_price: pd.DataFrame,
):

    pt = gpd.GeoDataFrame(
        {"lat": [lat], "lon": [lon]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )

    hit = gpd.sjoin(pt, districts_gdf, how="left", predicate="within")
    if hit.empty or hit["district"].isna().all():
        raise ValueError(
            "Point does not fall within any district polygon (check CRS/bounds)."
        )
    district = hit["district"].iloc[0]

    cov = district_coverage.rename(columns={"district": "district_name"})
    lp = land_price.rename(columns={"district": "district_name"})
    ctx = (
        pd.DataFrame({"district_name": [district], "lat": [lat], "lon": [lon]})
        .merge(cov, on="district_name", how="left")
        .merge(lp, on="district_name", how="left")
    )
    return ctx


def calculate_new_site_distances(new_site, df_sites):
    distances = []
    for _, row in df_sites.iterrows():
        d = haversine_km(
            new_site["lat"], new_site["lon"], row["latitude"], row["longitude"]
        )
        distances.append(
            {"from_site_id": "new", "to_site_id": row["site_id"], "distance": d}
        )
    return pd.DataFrame(distances)


def build_df_place_to_predict_knn(new_site_info, num_of_chargers, charger_type):
    """
    Build a single-row DataFrame for a new site, ready for KNN total-volume prediction.
    
    new_site_info: pd.DataFrame or dict with columns like population, total_volume, etc.
    """
    features_needed = [
    'num_of_chargers', 'population', 'lat', 'lon',
    'total_volume', 'volume_per_capita', 'sites_per_100k', 'underserved_index', 'price'
]
    data = {col: new_site_info[col].iloc[0] if col in new_site_info else 0 for col in features_needed}

    data['num_of_chargers'] = num_of_chargers
    data['charger_type'] = charger_type
    data['site_id'] = new_site_info.get('site_id', 'new') if isinstance(new_site_info, dict) else 'new'
    rated_power_map = {
    'AC_slow': 7,
    'AC_fast': 22,
    'DC_slow': 60,
    'DC_fast': 150,
    'Ultra_fast': 360
}
    data['avg_power'] = place_3_df['charger_type'].map(rated_power_map)[0]
    

    # Create single-row DataFrame
    place_df = pd.DataFrame([data])
    return place_df


def poi_counts_within_2km(place_x_df, poi_df, radius_km=2.0):
    # place_x must have columns 'lat' and 'lon'
    sites_rad = np.radians(place_x_df[['lat','lon']].values)  # shape (n_sites, 2)
    poi_rad = np.radians(poi_df[['latitude','longitude']].values)  # shape (n_pois, 2)
    
    tree = BallTree(poi_rad, metric='haversine')
    r = radius_km / 6371.0  # km -> radians
    ind = tree.query_radius(sites_rad, r=r)
    
    counts = np.array([len(ix) for ix in ind])
    
    # Return a DataFrame
    out = pd.DataFrame({
        'site_id': place_x_df['site_id'].values,
        'poi_within_2km': counts
    })
    return out

def predict_new_site_total_volume_knn(place_df, gdf_full_updated, distance_long,
                                      place_x=None,
                                      effective_radius_km=None,
                                      k_neighbors=None,
                                      power_weight=None,
                                      charger_scaling_exp=None):
    """
    Predict 6-month total volume using KNN on nearby sites,
    including district-level features from 'district_info' or 'gdf_full_updated'.
    """
    # --- 1. Filter neighbors within radius ---
    nearby = distance_long[distance_long['from_site_id'] == 'new'].copy()
    
    # Merge site info from gdf_full_updated
    nearby = nearby.merge(
        gdf_full_updated[['site_id', 'total_volume', 'num_of_chargers', 'charger_type',
            'population', 'poi_within_2km', 'avg_power',
             'district'
        ]],
        left_on='to_site_id', right_on='site_id', how='left'
    )
    nearby = nearby[nearby['distance'] <= effective_radius_km]


    nearby = nearby.merge(
            place_x[['district_name','volume_per_capita','sites_per_100k','underserved_index']],
            left_on='district', right_on='district_name', how='left'
        )
    for col in ['volume_per_capita', 'sites_per_100k', 'underserved_index']:
        nearby[col] = nearby[col].fillna(nearby[col].median())
    nearby = nearby.sort_values(by="distance").groupby("site_id", as_index=False).first()

    # --- 3. Features for KNN ---
    features = ['num_of_chargers','population','poi_within_2km','volume_per_capita','sites_per_100k','underserved_index', 'avg_power']
    X = nearby[features].values
    y = nearby['total_volume'].values

    # --- 4. Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_X_scaled = scaler.transform(place_df[features].values)

    # --- 5. Fit KNN ---
    knn = KNeighborsRegressor(n_neighbors=min(k_neighbors, len(nearby)))
    knn.fit(X_scaled, y)
    
    distances, indices = knn.kneighbors(new_X_scaled)
    neighbor_info = nearby.iloc[indices[0]].copy()
    neighbor_info["distance_in_feature_space"] = distances[0]
    neighbor_info["weight"] = 1 / (distances[0] + 1e-6)
    
    pred_total = knn.predict(new_X_scaled)[0]

    # --- 6. Scale by new site's chargers ---
    predicted_volume = pred_total
    # after pred_total is computed
    delta_ratio = (place_df['avg_power'].values[0] - neighbor_info['avg_power'].mean()) / neighbor_info['avg_power'].mean()
    delta_ratio = np.clip(delta_ratio, -0.5, 0.5)  # max ±50% adjustment
    pred_total_adjusted = pred_total * (1 + power_weight * delta_ratio)

# then scale by chargers
    predicted_volume = pred_total_adjusted * place_df['num_of_chargers'].values[0] ** charger_scaling_exp
    print("KNN raw prediction (before scaling):", pred_total)
    print("Num chargers:", place_df['num_of_chargers'].values[0])
    print("Scaling exponent:", charger_scaling_exp)

    return predicted_volume, pd.DataFrame(neighbor_info), list(features)

def predict_new_site_total_volume_knn_weighted(
    place_df, gdf_full_updated, distance_long,
    place_x=None,
    effective_radius_km=6,
    k_neighbors=10,
    charger_scaling_exp=0.7,
    power_weight=0.6,           # fraction of influence for avg_power
    weight_same_type=0.7         # weight for neighbors with same charger type
):
    """
    Predict 6-month total volume using KNN, including partial weighting for avg_power
    and optional emphasis on neighbors with the same charger type.
    """
    # --- 1. Filter neighbors within radius ---
    nearby = distance_long[distance_long['from_site_id'] == 'new'].copy()
    nearby = nearby.merge(
        gdf_full_updated[['site_id','total_volume','num_of_chargers','charger_type',
                          'population','poi_within_2km','avg_power','district']],
        left_on='to_site_id', right_on='site_id', how='inner'
    )
    nearby = nearby[nearby['distance'] <= effective_radius_km]

    if nearby.empty:
        # fallback: average volume per charger
        avg_volume_per_charger = gdf_full_updated['total_volume'].sum() / gdf_full_updated['num_of_chargers'].sum()
        return avg_volume_per_charger * place_df['num_of_chargers'].values[0]**charger_scaling_exp

    # --- 2. Merge district-level features if place_x provided ---
    if place_x is not None:
        nearby = nearby.merge(
            place_x[['district_name','volume_per_capita','sites_per_100k','underserved_index']],
            left_on='district', right_on='district_name', how='left'
        )
    for col in ['volume_per_capita','sites_per_100k','underserved_index']:
        nearby[col] = nearby[col].fillna(nearby[col].median())

    nearby = nearby.sort_values(by="distance").groupby("site_id", as_index=False).first()

    # --- 3. Features for KNN ---
    features = ['num_of_chargers','population','poi_within_2km',
                'volume_per_capita','sites_per_100k','underserved_index','avg_power']
    X = nearby[features].values
    y = nearby['total_volume'].values
    types = nearby['charger_type'].values

    # --- 4. Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    new_X_scaled = scaler.transform(place_df[features].values)

    # --- 5. Reduce avg_power influence ---
    power_idx = features.index('avg_power')
    X_scaled[:, power_idx] *= power_weight
    new_X_scaled[:, power_idx] *= power_weight

    # --- 6. Compute KNN ---
    knn = KNeighborsRegressor(n_neighbors=min(k_neighbors, len(nearby)))
    knn.fit(X_scaled, y)
    distances, indices = knn.kneighbors(new_X_scaled)
    neighbor_info = nearby.iloc[indices[0]].copy()
    neighbor_info["distance_in_feature_space"] = distances[0]
    neighbor_info["weight"] = 1 / (distances[0] + 1e-6)

    # --- 7. Apply same-type weighting ---
    same_type_mask = neighbor_info['charger_type'] == place_df['charger_type'].values[0]
    neighbor_info.loc[same_type_mask, 'weight'] *= (1 + weight_same_type)

    # Normalize weights
    neighbor_info['weight'] /= neighbor_info['weight'].sum()  # normalize
    pred_total = (neighbor_info['total_volume'] * neighbor_info['weight']).sum()
    
    # --- 7. Apply soft avg_power adjustment ---
    avg_neighbor_power = neighbor_info['avg_power'].mean()
    delta_ratio = (place_df['avg_power'].values[0] - avg_neighbor_power) / avg_neighbor_power
    delta_ratio = np.clip(delta_ratio, -0.5, 0.5)  # limit ±50% effect
    
    pred_total = pred_total * (1 + power_weight * delta_ratio)
    
    # --- 8. Apply charger scaling ---
    predicted_volume = pred_total * place_df['num_of_chargers'].values[0]**charger_scaling_exp

    return predicted_volume, neighbor_info.reset_index(drop=True), features

def estimate_site_investment(
    district: str,
    area_sqm: float,
    charger_type: str,
    num_chargers: int,
    land_price: pd.DataFrame,  # columns: ['district','price_per_m2']
    chargers_market_price: pd.DataFrame,  # {'AC_slow':5000, ...}
    misc_costs: pd.DataFrame,  # {'installation':6000, 'grid_connection':2000, 'site_preparation_and_base':1000}
    misc_split=(
        "per_charger",
        {"installation", "site_preparation_and_base"},
        {
            "grid_connection",
        },
    ),  # (mode, per_charger_set, per_site_set)
    contingency_pct: float = 0.05,  # 5% contingency
) -> dict:
    """Return a breakdown dict with totals and each component."""
    # --- land ---
    row = land_price.loc[
        land_price["district"].str.lower().str.strip() == district.lower().strip()
    ]
    if row.empty:
        raise ValueError(f"District '{district}' not found in land_price_df.")
    price_per_m2 = float(row["price"].iloc[0])
    land_cost = area_sqm * price_per_m2

    # --- hardware ---
    charger_row = chargers_market_price.loc[
        chargers_market_price["charger_type"].str.lower().str.strip()
        == charger_type.lower().strip()
    ]
    if charger_row.empty:
        raise ValueError(
            f"Charger type '{charger_type}' not found in chargers_market_price."
        )
    unit_price = float(charger_row["price"].iloc[0])
    hardware_cost = num_chargers * unit_price

    # --- misc ---
    misc_costs_dict = dict(
        zip(misc_costs["service"].str.lower().str.strip(), misc_costs["price"])
    )

    per_charger_keys = {k.lower().strip() for k in misc_split[1]}
    per_site_keys = {k.lower().strip() for k in misc_split[2]}

    misc_per_charger_sum = sum(misc_costs_dict.get(k, 0) for k in per_charger_keys)
    misc_per_site_sum = sum(misc_costs_dict.get(k, 0) for k in per_site_keys)
    misc_cost = num_chargers * misc_per_charger_sum + misc_per_site_sum

    # --- subtotal & contingency ---
    subtotal = land_cost + hardware_cost + misc_cost
    contingency = contingency_pct * subtotal
    total = subtotal + contingency

    return {
        "district": district,
        "area_sqm": area_sqm,
        "charger_type": charger_type,
        "num_chargers": num_chargers,
        "price_per_m2": price_per_m2,
        "unit_price": unit_price,
        "land_cost": land_cost,
        "hardware_cost": hardware_cost,
        "misc_cost": misc_cost,
        "contingency_pct": contingency_pct,
        "contingency": contingency,
        "total_investment": total,
    }

def calculate_ROI(place_x, place_x_df, gdf_full_updated, total_investment_place_x, predicted_volume_place_x):

    place_x_static = {
    'district_name': place_x['district_name'].iloc[0],
    'lat': place_x_df['lat'].iloc[0],
    'lon': place_x_df['lon'].iloc[0],
    'charger_type': place_x_df['charger_type'].iloc[0],
    'num_of_chargers': place_x_df['num_of_chargers'].iloc[0]
}
    place_x_charger_type = place_x_df['charger_type'][0]
    
    electricity_prices = gdf_full_updated[gdf_full_updated['charger_type'] == place_x_charger_type] \
            .agg({'e_price': 'mean', 's_price': 'mean'}).to_dict()

    place_x_performance = {**place_x_static, **electricity_prices, 'charger_type': place_x_df['charger_type'][0]}
    place_x_performance_df = pd.DataFrame([place_x_performance])
    
    place_x_performance_df['total_volume'] = predicted_volume_place_x

    place_x_performance_df['cost'] = place_x_performance_df['total_volume'] * place_x_performance_df['e_price']           
    place_x_performance_df['profit'] = place_x_performance_df['total_volume'] * place_x_performance_df['s_price']            
    place_x_performance_df['revenue'] = place_x_performance_df['cost'] + place_x_performance_df['profit'] 

    months_to_ROI_place_x = total_investment_place_x / place_x_performance_df['profit'] * 6

    return f"{months_to_ROI_place_x[0].round().astype(int)} months to ROI"

    