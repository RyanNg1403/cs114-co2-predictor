async def get_recommendations(features, predictions, agent):
    """
    Generate recommendations based on input features and predictions using Gemini.
    
    Args:
        features (dict): Dictionary of input features
        predictions (dict): Dictionary of model predictions (only Lasso CD)
        
    Returns:
        str: Generated recommendations
    """
    # Create a comprehensive system prompt
    system_prompt = """You are an AI assistant specialized in providing recommendations for reducing CO2 emissions based on agricultural and food production data. 
    Your task is to analyze the input features and Lasso CD predictions to provide actionable, specific recommendations.
    
    Focus on:
    1. Identifying key factors contributing to high emissions
    2. Suggesting practical, implementable changes
    3. Prioritizing recommendations based on impact
    4. Providing context-specific advice based on the country/area
    5. Make sure to include the value of the feature you are recommending to change. For example, when recommending solutions to reduce Average temperature, mention the current average temperature.
    
    Format your response in clear sections with bullet points."""
    
    # Create the user prompt with relevant information
    user_prompt = f"""Based on the following data, provide specific recommendations for reducing CO2 emissions:

Input Features:
{format_features(features)}

Predicted CO2 Emissions (Lasso CD):
{format_predictions(predictions)}

Please provide actionable recommendations focusing on the most impactful changes first."""

    # Get recommendations from Gemini
    result = await agent.run(system_prompt + "\n\n" + user_prompt)
    return result.output

def format_features(features):
    """Format features for the prompt."""
    formatted = []
    for key, value in features.items():
        if key == 'Area':
            formatted.append(f"Country/Area: {value}")
        elif 'population' in key.lower() or "year" in key.lower():
            formatted.append(f"{key}: {int(value)}")
        else:
            formatted.append(f"{key}: {value:.2f}")
    return "\n".join(formatted)

def format_predictions(predictions):
    """Format predictions for the prompt."""
    return "\n".join([f"{model}: {pred:.2f}" for model, pred in predictions.items()]) 