from groq import Groq

def generate_recipes(ingredients: str, api_key: str):
    """
    Backend function to generate recipes using Groq API.
    """

    client = Groq(api_key=api_key)

    prompt = f"""
    You are a professional Indian home chef.

    Using ONLY these ingredients:
    {ingredients}

    Generate exactly 2 recipes.

    Each recipe must include:
    - Title
    - Total time (in minutes)
    - Ingredient list
    - Step-by-step instructions
    - Veg-friendly unless ingredients suggest otherwise
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
