import streamlit as st
from src.objective4.backend.recipe_generator import generate_recipes


def recipe_ui():
    st.header("🍽 Recipe Generator from Ingredients")

    ingredients = st.text_area(
        "Enter the food items you have:",
        placeholder="Example: rice, tomato, onion, capsicum, leftover dal"
    )

    if st.button("Generate Recipes"):
        if not ingredients.strip():
            st.error("Please enter at least one ingredient.")
            return

        # Read API key
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception as e:
            st.error(f"Could not read GROQ_API_KEY: {e}")
            return

        # Backend call
        try:
            result = generate_recipes(ingredients, api_key)
            st.success("✅ Here are your recipes:")
            st.write(result)

        except Exception as e:
            st.error(f"Error generating recipes: {e}")


def main():
    st.title("🍽 Food Waste Reduction System")
    st.sidebar.title("Navigation")

    page = st.sidebar.selectbox(
        "Choose Objective",
        ["Recipe Generator"]
    )

    if page == "Recipe Generator":
        recipe_ui()


if __name__ == "__main__":
    main()


