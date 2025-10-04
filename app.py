import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv
import googlemaps

# Load Google Maps API key
load_dotenv()
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

# Import your project utils
from utils import (
    attach_site_context, haversine_km, calculate_new_site_distances,
    build_df_place_to_predict_knn, poi_counts_within_2km,
    predict_new_site_total_volume_knn, predict_new_site_total_volume_knn_weighted,
    estimate_site_investment, calculate_ROI
)

st.set_page_config(
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load saved variables
with open("saved_vars.pkl", "rb") as f:
    data = pickle.load(f)

df_daily = data["df_daily"]
district_coverage = data["district_coverage"]
miscellaneous_costs = data["miscellaneous_costs"]
chargers_market_price = data["chargers_market_price"]
chargers_gdf = data["chargers_gdf"]
gdf_full_updated = data["gdf_full_updated"]
land_price = data["land_price"]
poi_per_district = data["poi_per_district"]
districts_gdf = data["districts_gdf"]
distance_long = data["distance_long"]
unique_sites = data["unique_sites"]

menu = st.sidebar.radio("Select Page", ["Find Potential New Site", "🔍 About Us"])

if menu == "Find Potential New Site":
        # --- Initialize session state ---
    if "location" not in st.session_state:
        st.session_state.location = None  # {'lat', 'lon', 'source'}
    if "searched_address" not in st.session_state:
        st.session_state.searched_address = ""  # Stores formatted address from search

    st.title("Find Potential New Site")
    st.write("Use the form below to find potential new sites for your business.")

    # Bounding box (Shenzhen example)
    lat_min, lat_max = 22.46557, 22.818918
    lon_min, lon_max = 113.784724, 114.516991
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # --- Address search ---
    st.write("### Select site location inside city boundary")
    st.write("You can click on the map or search by address:")

    address_input = st.text_input("Street, number, city (e.g. Coco Park, Shenzhen)")

    if st.button("Search") and address_input:
        results = gmaps.geocode(address_input, language="zh-CN")
        if results:
            lat = results[0]["geometry"]["location"]["lat"]
            lon = results[0]["geometry"]["location"]["lng"]
            formatted = results[0]["formatted_address"]

            # Save in session state
            st.session_state.location = {"lat": lat, "lon": lon, "source": "search"}
            st.session_state.searched_address = formatted
        else:
            st.error("⚠️ Address not found.")

    st.write("")
    st.markdown(
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔵 Blue: Existing sites&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🟡 Yellow: Points of interest (POIs)",
    unsafe_allow_html=True
)

    # --- Build folium map ---
    map_center = [st.session_state.location["lat"], st.session_state.location["lon"]] if st.session_state.location else [center_lat, center_lon]
    zoom = 15 if st.session_state.location else 11
    map_szx = folium.Map(location=map_center, zoom_start=zoom)

    # Draw bounding box
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="blue", fill=True, fill_opacity=0.1
    ).add_to(map_szx)

    # Add sites (blue)
    for _, row in unique_sites.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2, color="blue", fill=True, fill_opacity=0.6
        ).add_to(map_szx)

    # Add POIs (yellow)
    for _, row in poi_per_district.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2, color="yellow", fill=True, fill_opacity=0.6
        ).add_to(map_szx)

    # Add red marker for selected location
    if st.session_state.location:
        folium.Marker(
            location=[st.session_state.location["lat"], st.session_state.location["lon"]],
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(map_szx)

    # Enable click popup
    map_szx.add_child(folium.LatLngPopup())

    # Render map
    map_data = st_folium(map_szx, width=700, height=500)

    # Update session state if user clicks on map
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        # Reverse geocode to get formatted address
        results = gmaps.reverse_geocode((clicked_lat, clicked_lon), language="zh-CN")
        if results:
            formatted_address = results[0]["formatted_address"]
        else:
            formatted_address = "Address not found"

    # Save to session state
        st.session_state.location = {"lat": clicked_lat, "lon": clicked_lon, "source": "click"}
        st.session_state.searched_address = formatted_address

    if st.session_state.location and st.session_state.searched_address:
        st.success(f"📍 Last selected location: {st.session_state.searched_address} → "
               f"{st.session_state.location['lat']}, {st.session_state.location['lon']}")
