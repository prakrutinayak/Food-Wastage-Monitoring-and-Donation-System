import streamlit as st
import pandas as pd
# Import functions from the logic file
from food_aid_logic import (
    load_ngo_data, 
    get_supermarket_data, 
    get_initial_demands, 
    calculate_metrics, 
    get_demands_for_supermarket, 
    create_new_demand,
    calculate_distance # Included here as it might be useful for front-end debugging/display
)

# --- 1. Initial Data Setup (Initialize State) ---

def initialize_data():
    """Loads data into Streamlit session state."""
    # --- Load NGO Data ---
    if 'ngo_df' not in st.session_state:
        df_ngo = load_ngo_data()
        st.session_state.ngo_df = df_ngo
        if df_ngo.empty:
             st.error("Error: NGO data could not be loaded. Check 'food_banks (1).csv'.")
             return
            
    # --- Simulated Supermarket Data ---
    if 'supermarkets' not in st.session_state:
        st.session_state.supermarkets = get_supermarket_data(st.session_state.ngo_df)
            
    # --- Simulated Demand Data ---
    if 'demands' not in st.session_state:
        st.session_state.demands = get_initial_demands()

# --- 2. Metrics and Tracking Dashboard Module ---

def metrics_dashboard():
    st.title("📊 Platform Performance Dashboard")
    
    if not st.session_state.demands:
        st.info("No demands have been entered yet to display metrics.")
        return

    metrics = calculate_metrics(st.session_state.demands)

    # 1. Demand Summary
    st.subheader("1. Demand Status Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Demands", metrics.get('total_demands', 0))
    col2.metric("🟢 Fulfilled/Available", metrics.get('fulfilled_count', 0))
    col3.metric("🟡 Pending Review", metrics.get('pending_count', 0))
    col4.metric("🔴 Out of Stock", metrics.get('out_of_stock_count', 0))

    st.markdown("---")

    # 2. Fulfillment Rate
    st.subheader("2. Demand Fulfillment Rate")
    
    fulfillment_rate = metrics.get('fulfillment_rate', 0)
    if metrics.get('total_demands', 0) > 0:
        st.progress(fulfillment_rate / 100)
        st.metric("Fulfillment Rate", f"{fulfillment_rate:.2f}%", help="Based on demands marked as 'Stock Available' or 'Partially Available'.")
    else:
        st.info("No demands to calculate fulfillment rate.")

    st.markdown("---")

    # 3. Top Needed Items
    st.subheader("3. Top Requested Items")
    
    df_top = metrics.get('top_items_df')
    if not df_top.empty:
        st.table(df_top.set_index('Item'))
        st.caption("Chart functionality has been replaced by a table to resolve Python 3.12 dependency errors.")
    else:
        st.info("No items recorded yet.")

# --- 3. NGO Interface Module ---
def ngo_interface():
    st.title("🤝 NGO Demand Entry")
    
    if st.session_state.ngo_df.empty:
        st.warning("NGO data not loaded. Please fix file path and restart.")
        return
    
    # Prepare NGO options for selection
    ngo_options = st.session_state.ngo_df.set_index('id')['name'].to_dict()
    
    # Select NGO (Default to the first one)
    selected_ngo_id = st.selectbox(
        "Select your NGO:", 
        list(ngo_options.keys()), 
        format_func=lambda x: ngo_options[x]
    )
    
    ngo_name = ngo_options[selected_ngo_id]
    
    st.subheader(f"Logged in as: **{ngo_name}**")
    
    with st.form("new_demand_form"):
        st.write("### New Food Product Demand")
        
        item_name = st.text_input("Item Name (e.g., Rice, Milk, Lentils)", "Canned Beans")
        quantity_needed = st.number_input("Quantity Needed", min_value=1.0, value=75.0)
        unit = st.selectbox("Unit", ['kg', 'bags', 'liters', 'packets'], index=3)
        
        submitted = st.form_submit_button("Submit Demand")
        
        if submitted:
            new_demand = create_new_demand(
                st.session_state.demands,
                selected_ngo_id,
                item_name,
                quantity_needed,
                unit
            )
            st.session_state.demands.append(new_demand)
            st.success(f"Demand for **{item_name}** submitted successfully for {ngo_name}!")
            st.experimental_rerun()

    st.markdown("---")
    st.subheader(f"Current Demand Status for {ngo_name}")
    
    # Filter demands by the selected NGO ID
    df_demands = pd.DataFrame([d for d in st.session_state.demands if d['ngo_id'] == selected_ngo_id])
    if not df_demands.empty:
        st.dataframe(df_demands[['item_name', 'quantity_needed', 'unit', 'status', 'supermarket_response']])
    else:
        st.info("No active demands.")

# --- 4. Supermarket Interface Module ---
def supermarket_interface():
    st.title("🛒 Supermarket Stock Management")
    
    if st.session_state.ngo_df.empty:
        st.warning("NGO data not loaded. Please fix file path and restart.")
        return
        
    sm_id = st.selectbox("Select Supermarket:", list(st.session_state.supermarkets.keys()), format_func=lambda x: st.session_state.supermarkets[x]['name'])
    sm_data = st.session_state.supermarkets[sm_id]
    st.subheader(f"Logged in as: **{sm_data['name']}**")
    
    st.markdown("---")
    st.subheader("Demands from Nearby NGOs")
    
    # Get demands filtered by proximity (logic handled in food_aid_logic.py)
    df_sm_view = get_demands_for_supermarket(
        sm_id, 
        sm_data, 
        st.session_state.demands, 
        st.session_state.ngo_df
    )
    
    if df_sm_view.empty:
        st.info("No nearby pending demands.")
        return

    st.dataframe(df_sm_view)
    
    st.markdown("---")
    st.subheader("Update Stock Status")
    
    pending_demand_ids = df_sm_view[df_sm_view['Status'] == 'Pending Supermarket Review'].index.tolist()
    
    if not pending_demand_ids:
        st.info("No demands require stock updates.")
        return
        
    selected_demand_id = st.selectbox("Select Demand to Update:", pending_demand_ids)
    
    if selected_demand_id:
        # Find the index in the *global* session state list
        demand_index = next(i for i, d in enumerate(st.session_state.demands) if d['id'] == selected_demand_id)
        current_demand = st.session_state.demands[demand_index]
        
        # Get NGO name for display
        ngo_name = st.session_state.ngo_df.set_index('id').loc[current_demand['ngo_id'], 'name']

        st.write(f"Updating: **{current_demand['item_name']}** ({current_demand['quantity_needed']} {current_demand['unit']}) for {ngo_name}")

        col1, col2, col3 = st.columns(3)
        changed = False

        if col1.button("✅ I Have It In Stock"):
            st.session_state.demands[demand_index]['status'] = 'Stock Available'
            st.session_state.demands[demand_index]['supermarket_response'] = sm_data['name']
            st.success(f"Status updated to **Stock Available**! NGO has been notified.")
            changed = True
        
        if col2.button("❌ Out of Stock"):
            st.session_state.demands[demand_index]['status'] = 'Out of Stock'
            st.session_state.demands[demand_index]['supermarket_response'] = sm_data['name']
            st.warning(f"Status updated to **Out of Stock**.")
            changed = True

        # Using st.expander for partial availability
        with col3.expander("⏳ Partially Available"):
            max_qty = current_demand['quantity_needed'] - 0.01 if current_demand['quantity_needed'] > 1 else current_demand['quantity_needed']
            
            partial_quantity = st.number_input(f"Quantity you can supply ({current_demand['unit']}):", 
                                                min_value=1.0, 
                                                max_value=max_qty, 
                                                value=current_demand['quantity_needed'] / 2 if current_demand['quantity_needed'] > 1 else 1.0,
                                                key=f"partial_qty_{selected_demand_id}")
            if st.button("Confirm Partial Supply", key=f"partial_btn_{selected_demand_id}"):
                st.session_state.demands[demand_index]['status'] = f"Partially Available ({partial_quantity} {current_demand['unit']})"
                st.session_state.demands[demand_index]['supermarket_response'] = sm_data['name']
                st.info(f"Status updated: Partial supply confirmed.")
                changed = True

        if changed:
            st.experimental_rerun()
            
# --- 5. Main Application Logic ---

def main():
    st.set_page_config(layout="wide")
    initialize_data()
    
    st.sidebar.title("App Navigation")
    
    mode = st.sidebar.radio("Select View:", ["Map Overview", "Metrics Dashboard", "NGO Interface", "Supermarket Interface"])
    
    # --- Global Map Overview ---
    if mode == "Map Overview":
        st.header("🌍 Resource Connection Map Overview")
        
        if st.session_state.ngo_df.empty:
            st.warning("Cannot display map: NGO data not loaded. Check file name and path.")
            return

        # Prepare NGO data for the map
        ngo_df = st.session_state.ngo_df[['latitude', 'longitude', 'name']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        ngo_df['Color'] = 'NGO' 

        # Prepare Supermarket data for the map
        sm_data_list = [v for k, v in st.session_state.supermarkets.items()]
        sm_df = pd.DataFrame(sm_data_list).rename(columns={'lat': 'lat', 'lon': 'lon'})
        sm_df['Color'] = 'Supermarket' 
        
        # Combine dataframes
        map_df = pd.concat([ngo_df, sm_df]).reset_index(drop=True)
        
        if not map_df.empty:
            st.map(map_df, latitude='lat', longitude='lon', zoom=12)
            
            st.markdown(
                """
                * **Blue dots** represent **NGOs** (from your CSV file)
                * **Red dots** represent **Supermarkets** (simulated nearby locations)
                """
            )
        st.markdown("---")

    # --- Interface Switching ---
    elif mode == "Metrics Dashboard":
        metrics_dashboard()
    elif mode == "NGO Interface":
        ngo_interface()
    elif mode == "Supermarket Interface":
        supermarket_interface()

if __name__ == '__main__':
    main()