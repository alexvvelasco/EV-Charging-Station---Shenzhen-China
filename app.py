import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv
import googlemaps
import streamlit as st
import streamlit.components.v1 as components



# Load Google Maps API key
load_dotenv()
GOOGLE_KEY = None

try:
    GOOGLE_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
except Exception:
    GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_KEY:
    raise ValueError("❌ Google Maps API key not found in secrets.toml or .env file.")

# --- Initialize Google Maps client ---
gmaps = googlemaps.Client(key=GOOGLE_KEY)

from utils import (
    attach_site_context, haversine_km, calculate_new_site_distances,
    build_df_place_to_predict_knn, poi_counts_within_2km,
    predict_new_site_total_volume_knn, predict_new_site_total_volume_knn_weighted,
    estimate_site_investment, calculate_ROI, calculate_ROI2
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
    for key, default in [
        ("location", None),
        ("searched_address", ""),
        ("show_summary", False),
        ("features_done", False),
        ("volume_done", False),
        ("investment_done", False),
        ("roi_done", False),
        ("place_df", None),
        ("place_distances", None),
        ("district_for_place", None),
        ("predicted_volume_place", None),
        ("pred_volume_place_weighted", None),
        ("total_inv", None),
        ("months_to_ROI", None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.title("Find Potential New Site")
    st.write("Use the form below to find potential new sites for your business.")

    st.subheader("You can use AI to help you find the best location!")
    st.title("EV Charging Station Site Selection AI")

    components.iframe(
        "https://project-spark-61.vercel.app/ev-chatbot",
        height=700,
        scrolling=True
    )

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
        results = gmaps.reverse_geocode((clicked_lat, clicked_lon), language="zh-CN")
        formatted_address = results[0]["formatted_address"] if results else "Address not found"
        st.session_state.location = {"lat": clicked_lat, "lon": clicked_lon, "source": "click"}
        st.session_state.searched_address = formatted_address

    if st.session_state.location and st.session_state.searched_address:
        st.success(f"📍 Last selected location: {st.session_state.searched_address} → "
               f"{st.session_state.location['lat']}, {st.session_state.location['lon']}")

    # If no location selected, prompt user to enter manually
    if st.session_state.location is None:
        st.warning("No location selected. Please enter latitude and longitude manually below.")
        manual_lat = st.number_input("Latitude", min_value=22.46557, max_value=22.818918, value=22.554307, step=0.001)
        manual_lon = st.number_input("Longitude", min_value=113.784724, max_value=114.516991, value=114.037954, step=0.001)
        if st.button("Set Manual Location"):
            st.session_state.location = {"lat": manual_lat, "lon": manual_lon, "source": "manual"}
            st.session_state.searched_address = "Manual entry"
            st.rerun()

    # --- Main workflow ---
    elif st.session_state.location:
        lat = st.session_state.location["lat"]
        lon = st.session_state.location["lon"]

        # Find district for selected location
        try:
            site_ctx = attach_site_context(
                lat=lat,
                lon=lon,
                districts_gdf=districts_gdf,
                district_coverage=district_coverage,
                land_price=land_price,
            )
            district = site_ctx["district_name"].iloc[0]
            st.info(f"District detected: **{district}**")
        except Exception as e:
            st.error(f"Could not determine district: {e}")
            district = None

        # --- Site Parameters ---
        st.write("### Site Parameters")
        area_sqm = st.slider("Area (sqm)", min_value=20, max_value=200, value=50, step=10)
        ordered_charger_types = ["AC_slow", "AC_fast", "DC_slow", "DC_fast", "Ultra_fast"]
        charger_types = [ct for ct in ordered_charger_types if ct in gdf_full_updated["charger_type"].unique()]
        charger_type = st.selectbox("Charger Type", charger_types)
        num_chargers = st.slider("Number of chargers", min_value=1, max_value=10, value=2, step=1)

        # --- Step 1: Show Selection Summary ---
        if st.button("Show Selection Summary") or st.session_state.show_summary:
            st.session_state.show_summary = True
            dict = {
                "Latitude": lat,
                "Longitude": lon,
                "District": district,
                "Area (sqm)": area_sqm,
                "Charger Type": charger_type,
                "Number of Chargers": num_chargers
            }
            summary = pd.DataFrame.from_dict([dict])
            st.write("")
            st.write(summary)
            st.write("")
            st.write("")

        # --- Step 2: Feature Engineering ---
        if st.session_state.show_summary:
            if st.button("Feature Engineering to build DataFrame") or st.session_state.features_done:
                if not st.session_state.features_done:
                    st.toast("Starting feature engineering...")
                    place_distances = calculate_new_site_distances(site_ctx, df_sites=gdf_full_updated)
                    place_distances = place_distances.drop_duplicates(subset=["to_site_id"])
                    place_distances["distance"] = place_distances["distance"].astype(str)
                    place_distances["distance"] = place_distances["distance"].str.split().str[-3]
                    place_distances["distance"] = place_distances["distance"].astype(float)
                    district_for_place = site_ctx["district_name"].iloc[0]
                    place_df = build_df_place_to_predict_knn(site_ctx, num_of_chargers=num_chargers, charger_type=charger_type)
                    poi_counts_place = poi_counts_within_2km(place_df, poi_per_district, radius_km=2.0)
                    place_df['poi_within_2km'] = poi_counts_place['poi_within_2km']
                    st.session_state.place_df = place_df
                    st.session_state.place_distances = place_distances
                    st.session_state.district_for_place = district_for_place
                    st.session_state.features_done = True
                    st.toast("Feature engineering completed.")
                st.success("Features for new site constructed.")
                st.write("### Final feature set for prediction:")
                st.dataframe(st.session_state.place_df.head())
                st.write("")
                st.write("")

        # --- Step 3: Predict Annual Volume ---
        if st.session_state.features_done:
            if st.button("Predict Annual Volume") or st.session_state.volume_done:
                if not st.session_state.volume_done:
                    st.toast("Starting volume prediction...")
                    predicted_volume_place, neighbor_info_place, features_place = predict_new_site_total_volume_knn(
                        place_df=st.session_state.place_df,
                        gdf_full_updated=gdf_full_updated,
                        distance_long=st.session_state.place_distances,
                        place_x=site_ctx,
                        effective_radius_km=3,
                        k_neighbors=40,
                        power_weight=0.7,
                        charger_scaling_exp=0.7
                    )
                    pred_volume_place_weighted, neighbor_info, features_used = predict_new_site_total_volume_knn_weighted(
                        place_df=st.session_state.place_df,
                        gdf_full_updated=gdf_full_updated,
                        distance_long=st.session_state.place_distances,
                        place_x=site_ctx,
                        effective_radius_km=3,
                        k_neighbors=40,
                        charger_scaling_exp=0.6,
                        power_weight=0.7,
                        weight_same_type=0.7
                    )
                    st.session_state.predicted_volume_place = predicted_volume_place
                    st.session_state.pred_volume_place_weighted = pred_volume_place_weighted
                    st.session_state.volume_done = True
                    st.toast("Volume prediction completed.")
                st.write(f"Predicted first 6 months Volume (ramp-up): **{st.session_state.predicted_volume_place:,.0f} kWh**")
                st.write(f"Predicted next 6 months Volume (after ramp-up): **{st.session_state.pred_volume_place_weighted:,.0f} kWh**")
                total_volume_first_year = st.session_state.predicted_volume_place + st.session_state.pred_volume_place_weighted
                st.success(f"Estimated Annual Volume first year (kWh): **{total_volume_first_year:,.0f} kWh**")
                total_volume_after_year1 = st.session_state.pred_volume_place_weighted * 2
                st.success(f"Estimated Annual Volume after year 1 (kWh): **{total_volume_after_year1:,.0f} kWh**")

        # --- Step 4: Estimate Investment ---
        if st.session_state.volume_done:
            if st.button("Estimate Investment") or st.session_state.investment_done:
                if not st.session_state.investment_done:
                    st.toast("Estimating investment...")
                    inv = estimate_site_investment(
                        district=st.session_state.district_for_place,
                        area_sqm=area_sqm,
                        charger_type=charger_type,
                        num_chargers=num_chargers,
                        land_price=land_price,
                        chargers_market_price=chargers_market_price,
                        misc_costs=miscellaneous_costs,
                    )
                    st.session_state.total_inv = inv["total_investment"]
                    st.session_state.investment_done = True
                    st.toast("Investment estimation completed.")
                st.success(f"#### Estimated Total Investment: **{st.session_state.total_inv:,.0f} RMB**")
                st.write("")
                st.write("")

        # --- Step 5: Calculate ROI ---
        if st.session_state.investment_done:
            if st.button("Calculate ROI") or st.session_state.roi_done:
                if not st.session_state.roi_done:
                    st.toast("Calculating ROI...")
                    months_to_ROI = calculate_ROI2(
                        site_ctx, st.session_state.place_df, gdf_full_updated,
                        st.session_state.total_inv,
                        st.session_state.predicted_volume_place,
                        st.session_state.pred_volume_place_weighted
                    )
                    st.session_state.months_to_ROI = months_to_ROI
                    st.session_state.roi_done = True
                    st.toast("ROI calculation completed.")
                st.success(f"#### Estimated ROI: **{st.session_state.months_to_ROI}**")
                st.write("")
                st.write("")
                st.info("Note: If you want to try another location, please refresh the page to reset the workflow.")
