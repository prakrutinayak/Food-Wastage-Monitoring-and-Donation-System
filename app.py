# app_ui.py
import streamlit as st
import pandas as pd
from backend import (
    load_dataset,
    find_nearby,
    geocode_address,
    geolocate_device,
    google_search_link,
    directions_distance_km,
)
from config import API_KEY
import traceback

st.set_page_config(page_title="Food Donation Recommendation", page_icon="🍱", layout="centered")
st.title("🍱 Food Donation Recommendation System")

st.write("""
This system helps *supermarkets, restaurants, and stores* find the nearest *food banks or NGOs*
to donate excess food. It uses Google Maps APIs (Geocoding, Geolocation, Distance Matrix) for high accuracy.
If no organizations are within the threshold, it suggests eco-friendly alternatives.
""")

# Load dataset (cached)
@st.cache_data(ttl=600)
def _load_data(path):
    return load_dataset(path)

DATASET_PATH = "food_banks.csv"
try:
    data = _load_data(DATASET_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ------------------------------- LOCATION INPUT -------------------------------
st.subheader("📍 Choose Location Input Method")
use_current = st.checkbox("Use My Current Location (auto-detect)")

user_lat = user_lng = None

if use_current:
    try:
        with st.spinner("Detecting your location..."):
            lat, lng, acc = geolocate_device(API_KEY)
            user_lat, user_lng = lat, lng
            st.success(f"✅ Auto-detected location (accuracy ~ {acc} meters).")
            st.info(f"Coordinates: {user_lat:.6f}, {user_lng:.6f}")
    except Exception as e:
        st.error("Auto-detection failed: " + str(e))

# Manual address input
if not user_lat:
    address = st.text_input(
        "Enter your full address or area (e.g., 6th Cross, Santosh Nagar, Bengaluru, Karnataka)",
        placeholder="Type your area or address..."
    )
    if address:
        try:
            with st.spinner("Geocoding address..."):
                lat, lng, formatted = geocode_address(address, API_KEY)
                user_lat, user_lng = lat, lng
                st.success(f"✅ Location found: {formatted}")
                st.info(f"Coordinates: {user_lat:.6f}, {user_lng:.6f}")
        except ValueError as ve:
            if str(ve) == "ZERO_RESULTS":
                st.warning("⚠️ Address not found. Try simplifying (area + city + state).")
            else:
                st.error("Geocoding failed: " + str(ve))
        except Exception as ex:
            st.error("Geocoding error: " + str(ex))

# -------------------------- FIND NEARBY NGOs --------------------------
if user_lat and user_lng:
    st.info("⏳ Calculating road distances to all NGOs... please wait.")

    try:
        nearby = find_nearby(data, user_lat, user_lng, API_KEY, radius_km=5.0)

        # Improve top 5 accuracy using Directions API
        if not nearby.empty:
            nearby = nearby.reset_index(drop=True)
            TOPN = min(5, len(nearby))
            for i in range(TOPN):
                row = nearby.loc[i]
                precise = directions_distance_km(
                    user_lat, user_lng, row["latitude"], row["longitude"], API_KEY
                )
                if precise is not None:
                    nearby.at[i, "distance_km"] = precise

        # -------------------- NO NEARBY NGOs --------------------
        if nearby.empty:
            st.warning("⚠️ No nearby food banks found within 5 km (road distance).")

            # Keep your original info box
            st.info("""
                💡 You can consider these **sustainable alternatives**:
                - ♻️ **Composting Centers**
                - 🐄 **Animal Feed Donation**
                - 🏡 **Community Kitchens**
            """)

            # Clean emoji headings + long URL links
            st.subheader("♻️ Composting Centers")
            st.write(google_search_link("composting centers", user_lat, user_lng))

            st.subheader("🐄 Animal Feed Donation")
            st.write(google_search_link("animal shelters", user_lat, user_lng))

            st.subheader("🏡 Community Kitchens")
            st.write(google_search_link("community kitchens", user_lat, user_lng))

        # -------------------- SHOW NEARBY NGOs --------------------
        else:
            st.success(f"✅ Found {len(nearby)} nearby NGOs within 5 km (road distance).")
            st.subheader("🏠 Nearest Food Banks / NGOs")
            st.dataframe(nearby[["name", "address", "phone", "website", "distance_km"]])

            st.subheader("🗺️ View on Map")
            st.map(nearby[["latitude", "longitude"]])

            st.subheader("🔗 Google Map Links & Directions")
            for _, row in nearby.iterrows():
                lat = row["latitude"]
                lng = row["longitude"]
                name = row["name"]
                website = row.get("website", "")

                # convert to clickable website
                if isinstance(website, str) and website.strip():
                    if not website.lower().startswith(("http://", "https://")):
                        website = "https://" + website.strip()
                else:
                    website = ""

                st.markdown(
                    f"📍 **[{name}]({website})**  \n"
                    f"🗺️ [Open in Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lng})  \n"
                    f"🚗 [Get Directions](https://www.google.com/maps/dir/?api=1&destination={lat},{lng}) — **{row['distance_km']} km**"
                )

    except Exception as e:
        st.error("Error while computing distances. See console for details.")
        st.write(traceback.format_exc())

st.info("✅ Application ready.")
