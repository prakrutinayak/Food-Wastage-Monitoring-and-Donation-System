# backend.py
import re
import time
import math
import logging
from functools import lru_cache
from typing import List, Tuple, Optional

import pandas as pd
import requests

# -----------------------
# Config / constants
# -----------------------
DEFAULT_CSV_PATH = "food_banks.csv"
PLACEID_CACHE_CSV = "food_banks_with_placeids.csv"  # output if you populate place_ids

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Helpers
# -----------------------
def clean_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9, \-]', ' ', address)
    return " ".join(cleaned.split())

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# Simple coordinate sanity check
def _coords_valid(lat, lng) -> bool:
    try:
        lat = float(lat); lng = float(lng)
    except Exception:
        return False
    return (-90 <= lat <= 90) and (-180 <= lng <= 180)

# -----------------------
# Dataset loader
# -----------------------
def load_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"name", "address", "latitude", "longitude", "phone", "website"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # ensure place_id column exists (optional)
    if "place_id" not in df.columns:
        df["place_id"] = None
    return df

# -----------------------
# Geocoding & Geolocation
# -----------------------
@lru_cache(maxsize=1024)
def geocode_address(address: str, api_key: str) -> Tuple[float, float, str]:
    if not address:
        raise ValueError("Empty address sent to geocode.")
    cleaned = clean_address(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={requests.utils.quote(cleaned)}&key={api_key}"
    r = requests.get(url, timeout=10).json()
    status = r.get("status")
    if status == "OK":
        loc = r["results"][0]["geometry"]["location"]
        formatted = r["results"][0].get("formatted_address", cleaned)
        return float(loc["lat"]), float(loc["lng"]), formatted
    elif status == "ZERO_RESULTS":
        raise ValueError("ZERO_RESULTS")
    else:
        raise RuntimeError(f"Geocoding API error: {status}")

def geolocate_device(api_key: str) -> Tuple[float, float, Optional[float]]:
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    r = requests.post(url, timeout=10).json()
    if "location" in r:
        lat = r["location"]["lat"]
        lng = r["location"]["lng"]
        acc = r.get("accuracy", None)
        return float(lat), float(lng), acc
    raise RuntimeError(f"Geolocation API failure: {r}")

# -----------------------
# Directions (precise route for one OD pair)
# -----------------------
def directions_distance_km(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float, api_key: str, mode: str = "driving") -> Optional[float]:
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin_lat},{origin_lng}"
        f"&destination={dest_lat},{dest_lng}"
        f"&mode={mode}"
        f"&units=metric"
        f"&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10).json()
    except Exception as e:
        logger.warning("Directions API request failed: %s", e)
        return None
    status = resp.get("status")
    if status != "OK":
        logger.debug("Directions API status: %s", status)
        return None
    try:
        legs = resp["routes"][0].get("legs", [])
        meters = sum(leg.get("distance", {}).get("value", 0) for leg in legs)
        return round(meters / 1000.0, 2)
    except Exception as e:
        logger.exception("Parsing Directions response failed: %s", e)
        return None

# -----------------------
# Batched Distance Matrix (multiple destinations)
# -----------------------
def get_road_distances_batch(origin_lat: float, origin_lng: float, dests: List[Tuple[float, float]], api_key: str, mode: str = "driving") -> List[Optional[float]]:
    """
    dests: list of (lat, lng). Returns distances in km (rounded) or None for each dest.
    Google allows multiple destinations in one request. Keep batch small ( <= 25 ).
    """
    if not dests:
        return []

    dest_param = "|".join(f"{lat},{lng}" for lat, lng in dests)
    url = (
        "https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={origin_lat},{origin_lng}"
        f"&destinations={dest_param}"
        f"&mode={mode}"
        f"&units=metric"
        f"&key={api_key}"
    )

    backoff = 1.0
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=10)
            resp = r.json()
        except Exception as e:
            logger.warning("DistanceMatrix request error (attempt %d): %s", attempt + 1, e)
            time.sleep(backoff)
            backoff *= 2
            continue

        top_status = resp.get("status")
        if top_status != "OK":
            logger.warning("DistanceMatrix top-level status: %s", top_status)
            if top_status in ("REQUEST_DENIED", "OVER_QUERY_LIMIT", "INVALID_REQUEST"):
                break
            time.sleep(backoff)
            backoff *= 2
            continue

        try:
            elements = resp["rows"][0]["elements"]
            out = []
            for el in elements:
                if el.get("status") == "OK" and "distance" in el:
                    meters = el["distance"]["value"]
                    out.append(round(meters / 1000.0, 2))
                else:
                    out.append(None)
            return out
        except Exception as e:
            logger.exception("Parsing DistanceMatrix response failed: %s", e)
            time.sleep(backoff)
            backoff *= 2
            continue

    logger.error("DistanceMatrix failed after retries. Returning None results.")
    return [None] * len(dests)

def compute_distances_batch(df: pd.DataFrame, user_lat: float, user_lng: float, api_key: str, batch_size: int = 20) -> pd.DataFrame:
    df2 = df.copy().reset_index(drop=True)
    coords = [(r["latitude"], r["longitude"]) for _, r in df2.iterrows()]
    results = []
    i = 0
    while i < len(coords):
        batch = coords[i:i+batch_size]
        # replace invalid coordinates with placeholders but keep order
        valid_batch = []
        valid_idx = []
        for idx, (lat, lng) in enumerate(batch):
            if _coords_valid(lat, lng):
                valid_batch.append((lat, lng))
            else:
                # keep placeholder so results length matches
                valid_batch.append((None, None))
        # call API only on valid ones, but build dest string that includes invalids as placeholders
        # simpler: call batch request with only valid coords and then map back — but to keep implementation straightforward
        # we'll call with the whole batch replacing invalids with a slight offset near origin to avoid api errors.
        call_batch = []
        for (lat, lng) in batch:
            if _coords_valid(lat, lng):
                call_batch.append((lat, lng))
            else:
                # place a placeholder 1000 km away to get a NOT_FOUND / no route -> returns element.status != OK
                call_batch.append((user_lat, user_lng))  # request will return zero distance; we will handle later as None
        distances = get_road_distances_batch(user_lat, user_lng, call_batch, api_key)
        # map distances: if original coords invalid mark None
        for (lat, lng), d in zip(batch, distances):
            if not _coords_valid(lat, lng):
                results.append(None)
            else:
                results.append(d)
        i += batch_size
        time.sleep(1.0)  # small pause to avoid QPS issues

    df2["distance_km"] = results[:len(df2)]
    # add haversine for sanity checks
    df2["haversine_km"] = df2.apply(
        lambda r: _haversine_km(user_lat, user_lng, float(r["latitude"]), float(r["longitude"])) if _coords_valid(r["latitude"], r["longitude"]) else None,
        axis=1
    )
    return df2

# -----------------------
# Places (find place_id) helpers
# -----------------------
def find_place_id_for_row(name: str, address: str, api_key: str) -> Tuple[Optional[str], dict]:
    text = f"{name} {address}"
    url = (
        "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        f"?input={requests.utils.quote(text)}"
        "&inputtype=textquery"
        "&fields=place_id,formatted_address,name,geometry"
        f"&key={api_key}"
    )
    try:
        r = requests.get(url, timeout=10).json()
    except Exception as e:
        logger.warning("Places find request failed: %s", e)
        return None, {}
    if r.get("status") == "OK" and r.get("candidates"):
        return r["candidates"][0].get("place_id"), r["candidates"][0]
    return None, r

def populate_place_ids(csv_path: str, api_key: str, out_csv: str = PLACEID_CACHE_CSV, pause: float = 0.5) -> pd.DataFrame:
    df = load_dataset(csv_path)
    if "place_id" not in df.columns:
        df["place_id"] = None
    for i, r in df.iterrows():
        if pd.notna(r.get("place_id")):
            continue
        pid, info = find_place_id_for_row(r["name"], r["address"], api_key)
        df.at[i, "place_id"] = pid
        time.sleep(pause)
    df.to_csv(out_csv, index=False)
    return df

# -----------------------
# Public find_nearby
# -----------------------
def find_nearby(df: pd.DataFrame, user_lat: float, user_lng: float, api_key: str, radius_km: float = 5.0, batch_size: int = 20) -> pd.DataFrame:
    """
    Returns nearby rows (distance_km <= radius_km), sorted by road distance.
    """
    df_with_dist = compute_distances_batch(df, user_lat, user_lng, api_key, batch_size=batch_size)
    # drop rows where we couldn't get road distance
    df_with_dist = df_with_dist.dropna(subset=["distance_km"]).copy()

    # sanity logging: warn if road < straight-line (rare)
    for _, row in df_with_dist.iterrows():
        try:
            if row["distance_km"] is not None and row["haversine_km"] is not None:
                if row["distance_km"] + 0.1 < row["haversine_km"]:
                    logger.warning("Road distance smaller than straight-line for %s: road=%s, straight=%s",
                                   row.get("name","?"), row["distance_km"], row["haversine_km"])
        except Exception:
            pass

    nearby = df_with_dist[df_with_dist["distance_km"] <= radius_km].sort_values("distance_km")
    return nearby

# -----------------------
# Utilities
# -----------------------
def google_search_link(query: str, lat: float = None, lng: float = None) -> str:
    q = requests.utils.quote(query)
    if lat is not None and lng is not None:
        return f"https://www.google.com/maps/search/?api=1&query={q}+near+{lat}%2C{lng}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"
