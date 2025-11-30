# backend.py
import pandas as pd
import requests
import time
import re
from functools import lru_cache
from typing import Optional

# Default dataset path (NOW EXPECTED INSIDE PROJECT FOLDER)
DEFAULT_CSV_PATH = "food_banks.csv"

# Helper: basic cleaning used before geocoding
def clean_address(address: str) -> str:
    """Remove unwanted symbols to help Google Maps read the address properly."""
    if not isinstance(address, str):
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9, \-]', '', address)
    return cleaned.strip()

def load_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load CSV and ensure expected columns exist."""
    df = pd.read_csv(csv_path)
    required = {"name", "address", "latitude", "longitude", "phone", "website"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df

@lru_cache(maxsize=1024)
def geocode_address(address: str, api_key: str):
    if not address:
        raise ValueError("Empty address sent to geocode.")
    cleaned = clean_address(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={requests.utils.quote(cleaned)}&key={api_key}"
    resp = requests.get(url).json()
    status = resp.get("status")
    if status == "OK":
        loc = resp["results"][0]["geometry"]["location"]
        formatted = resp["results"][0].get("formatted_address", cleaned)
        return float(loc["lat"]), float(loc["lng"]), formatted
    elif status == "ZERO_RESULTS":
        raise ValueError("ZERO_RESULTS")
    else:
        raise RuntimeError(f"Geocoding API error: {status}")

def geolocate_device(api_key: str):
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    resp = requests.post(url).json()
    if "location" in resp:
        lat = resp["location"]["lat"]
        lng = resp["location"]["lng"]
        acc = resp.get("accuracy", None)
        return float(lat), float(lng), acc
    else:
        raise RuntimeError(f"Geolocation API failure: {resp}")

def get_road_distance(origin_lat, origin_lng, dest_lat, dest_lng, api_key: str, mode="driving"):
    """
    Uses Distance Matrix API to get the road distance (meters -> km).
    Returns float km or None.
    """
    url = (
        "https://maps.googleapis.com/maps/api/distancematrix/json?"
        f"origins={origin_lat},{origin_lng}&destinations={dest_lat},{dest_lng}"
        f"&mode={mode}&units=metric&key={api_key}"
    )
    resp = requests.get(url).json()
    # brief pause to avoid very tight loops causing QPS issues
    time.sleep(0.18)
    try:
        element = resp["rows"][0]["elements"][0]
        if element.get("status") == "OK" and "distance" in element:
            meters = element["distance"]["value"]
            return round(meters / 1000.0, 2)
        else:
            return None
    except Exception:
        return None

def directions_distance_km(origin_lat, origin_lng, dest_lat, dest_lng, api_key: str, mode="driving") -> Optional[float]:
    """
    Uses Directions API for an authoritative driving distance (sums legs).
    Returns float km or None on error.
    """
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin_lat},{origin_lng}"
        f"&destination={dest_lat},{dest_lng}"
        f"&mode={mode}&units=metric&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10).json()
    except Exception:
        return None

    if resp.get("status") != "OK":
        return None

    try:
        # sum distances across all legs (usually one leg)
        legs = resp["routes"][0].get("legs", [])
        meters = sum(leg.get("distance", {}).get("value", 0) for leg in legs)
        return round(meters / 1000.0, 2)
    except Exception:
        return None

def compute_distances(df: pd.DataFrame, user_lat: float, user_lng: float, api_key: str):
    """
    Compute road distances using Distance Matrix for all rows.
    Returns a dataframe copy with distance_km column (may contain None).
    """
    df2 = df.copy()
    distances = []
    for _, row in df2.iterrows():
        lat = row["latitude"]
        lng = row["longitude"]
        if pd.isna(lat) or pd.isna(lng):
            distances.append(None)
            continue
        d = get_road_distance(user_lat, user_lng, lat, lng, api_key)
        distances.append(d)
    df2["distance_km"] = distances
    return df2

def find_nearby(df: pd.DataFrame, user_lat: float, user_lng: float, api_key: str, radius_km: float = 5.0):
    """
    Returns nearby rows (distance_km <= radius_km) sorted by distance_km.
    Note: If Distance Matrix returns None for a row, that row is dropped here.
    The app will perform a Directions fallback on top-N afterwards.
    """
    df_with_dist = compute_distances(df, user_lat, user_lng, api_key)
    df_with_dist = df_with_dist.dropna(subset=["distance_km"])
    nearby = df_with_dist[df_with_dist["distance_km"] <= radius_km].sort_values("distance_km")
    return nearby

def google_search_link(query: str, lat: float = None, lng: float = None):
    q = requests.utils.quote(query)
    if lat is not None and lng is not None:
        return f"https://www.google.com/maps/search/?api=1&query={q}+near+{lat}%2C{lng}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"
