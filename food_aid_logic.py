import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

# --- 1. Utility Functions (Distance Calculation) ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the earth 
    using the Haversine formula.
    """
    R = 6371  # Radius of Earth in kilometers

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# --- 2. Initial Data Setup (Load Data & Initialize State) ---

def load_ngo_data(file_name="food_banks.csv"):
    """Loads and cleans NGO data from a CSV file."""
    if os.path.exists(file_name):
        df_ngo_data = pd.read_csv(file_name)
        # Rename for map compatibility and drop rows with missing coordinates
        df_ngo_data = df_ngo_data[['name', 'latitude', 'longitude', 'address']].dropna(subset=['latitude', 'longitude'])
        
        # Ensure latitude and longitude are float types
        df_ngo_data['latitude'] = pd.to_numeric(df_ngo_data['latitude'], errors='coerce')
        df_ngo_data['longitude'] = pd.to_numeric(df_ngo_data['longitude'], errors='coerce')
        df_ngo_data.reset_index(drop=True, inplace=True)
        df_ngo_data['id'] = df_ngo_data.index + 1
        
        return df_ngo_data
    else:
        # Note: Streamlit-specific error handling should be in the frontend
        print(f"Error: NGO data file '{file_name}' not found.")
        return pd.DataFrame()

def get_supermarket_data(ngo_df):
    """Generates simulated supermarket data based on NGO data's average location."""
    if not ngo_df.empty:
        # Use average location of the NGOs for sensible proximity data
        avg_lat = ngo_df['latitude'].mean()
        avg_lon = ngo_df['longitude'].mean()
    else:
        # Fallback coordinates
        avg_lat = 12.95
        avg_lon = 77.65
            
    supermarkets = {
        101: {'name': 'SuperMart Central', 'lat': avg_lat + 0.005, 'lon': avg_lon - 0.005},
        102: {'name': 'Fresh Market East', 'lat': avg_lat + 0.015, 'lon': avg_lon + 0.010},
        103: {'name': 'Big Bazaar South', 'lat': avg_lat - 0.008, 'lon': avg_lon - 0.002}
    }
    return supermarkets

def get_initial_demands():
    """Returns the initial simulated demand data."""
    demands = [
        {'id': 1, 'ngo_id': 1, 'item_name': 'Rice (Basmati)', 'quantity_needed': 50.0, 'unit': 'kg', 'status': 'Pending Supermarket Review', 'supermarket_response': None},
        {'id': 2, 'ngo_id': 2, 'item_name': 'Milk (Long Life)', 'quantity_needed': 100.0, 'unit': 'liters', 'status': 'Pending Supermarket Review', 'supermarket_response': None},
        {'id': 3, 'ngo_id': 3, 'item_name': 'Cooking Oil', 'quantity_needed': 30.0, 'unit': 'liters', 'status': 'Pending Supermarket Review', 'supermarket_response': None},
        {'id': 4, 'ngo_id': 4, 'item_name': 'Lentils (Masoor)', 'quantity_needed': 75.0, 'unit': 'kg', 'status': 'Pending Supermarket Review', 'supermarket_response': None},
    ]
    return demands

# --- 3. Data Processing for Interfaces ---

def calculate_metrics(demands):
    """Calculates platform performance metrics from the demands list."""
    if not demands:
        return {}

    df_demands = pd.DataFrame(demands)
    total_demands = len(df_demands)
    
    status_counts = df_demands['status'].value_counts()
    
    # Safely get counts, accounting for "Partially Available (X units)" statuses
    fulfilled_available = status_counts.get('Stock Available', 0)
    partial_available = sum(1 for status in df_demands['status'] if 'Partially Available' in status)
    
    fulfilled_count = fulfilled_available + partial_available
    pending_count = status_counts.get('Pending Supermarket Review', 0)
    out_of_stock_count = status_counts.get('Out of Stock', 0)
    
    fulfillment_rate = (fulfilled_count / total_demands) * 100 if total_demands > 0 else 0
    
    # Calculate top requested items
    top_items = df_demands[df_demands['item_name'].astype(bool)]['item_name'].value_counts().nlargest(5)
    df_top = top_items.reset_index()
    df_top.columns = ['Item', 'Requests']
    
    return {
        'total_demands': total_demands,
        'fulfilled_count': fulfilled_count,
        'pending_count': pending_count,
        'out_of_stock_count': out_of_stock_count,
        'fulfillment_rate': fulfillment_rate,
        'top_items_df': df_top
    }

def get_demands_for_supermarket(sm_id, sm_data, demands, ngo_df):
    """Filters demands based on proximity to a selected supermarket."""
    
    if ngo_df.empty:
        return []

    # Map NGO IDs to their coordinates and name for quick lookup
    ngo_coords_map = ngo_df[['id', 'latitude', 'longitude', 'name']].set_index('id').T.to_dict('dict')
    
    demands_to_display = []
    
    for demand in demands:
        ngo_id = demand['ngo_id']
        if ngo_id not in ngo_coords_map: continue

        ngo_data = ngo_coords_map[ngo_id]
        
        distance = calculate_distance(
            sm_data['lat'], sm_data['lon'],
            ngo_data['latitude'], ngo_data['longitude']
        )
        
        # Filter within 10 km
        if distance < 10: 
            demands_to_display.append({
                'Demand ID': demand['id'],
                'NGO': ngo_data['name'],
                'Distance (km)': f"{distance:.2f}",
                'Item': demand['item_name'],
                'Quantity': f"{demand['quantity_needed']} {demand['unit']}",
                'Status': demand['status']
            })

    # Return as DataFrame for easy display in Streamlit
    if demands_to_display:
        df_sm_view = pd.DataFrame(demands_to_display).set_index('Demand ID')
        return df_sm_view
    else:
        return pd.DataFrame()


def create_new_demand(current_demands, selected_ngo_id, item_name, quantity_needed, unit):
    """Creates and returns a new demand dictionary."""
    new_demand = {
        'id': len(current_demands) + 1,
        'ngo_id': selected_ngo_id, 
        'item_name': item_name,
        'quantity_needed': quantity_needed,
        'unit': unit,
        'status': 'Pending Supermarket Review',
        'supermarket_response': None
    }
    return new_demand