import pickle
import pandas as pd
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Smart Weather App", page_icon="🌦️", layout="centered")

# ---------------- Load the trained ML model ----------------
# This model was trained on temp_max + wind -> weather (rain/sun/snow/drizzle/fog)
@st.cache_resource
def load_model():
    with open("weather_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- Helper: emojis for each weather type ----------------
WEATHER_EMOJI = {
    "sun": "☀️",
    "rain": "🌧️",
    "drizzle": "🌦️",
    "snow": "❄️",
    "fog": "🌫️",
}

# ---------------- Helper 1: Temperature Level ----------------
def get_temp_level(temp):
    if temp < 5:
        return "🥶 Very Cold"
    elif temp < 15:
        return "🧥 Cold"
    elif temp < 25:
        return "😊 Pleasant"
    elif temp < 32:
        return "🌤️ Warm"
    else:
        return "🔥 Hot"

# ---------------- Helper 2: What to Carry ----------------
def get_carry_list(weather, humidity, rain_chance):
    items = []

    # Base items depending on predicted weather
    base_tips = {
        "sun":     ["🕶️ Sunglasses", "🧴 Sunscreen", "🧢 A hat", "💧 Water bottle"],
        "rain":    ["☔ Umbrella", "🧥 Raincoat", "👢 Waterproof shoes"],
        "drizzle": ["🧥 Light jacket", "☂️ Small umbrella"],
        "snow":    ["🧤 Gloves", "🧣 Scarf", "🥾 Warm boots", "🧥 Heavy coat"],
        "fog":     ["🧥 Light jacket", "🔦 Extra caution if driving"],
    }
    items.extend(base_tips.get(weather, ["🎒 Your usual essentials"]))

    # Extra rules based on rain chance and humidity
    if rain_chance > 50 and "☔ Umbrella" not in items:
        items.append("☔ Umbrella (high rain chance)")
    if humidity > 70:
        items.append("💧 Extra water (humid air feels hotter)")

    return items

# ---------------- Helper 3: Health Recommendations ----------------
def get_health_tips(temp, humidity, wind, rain_chance):
    tips = []

    if temp > 30 and humidity > 60:
        tips.append("🥵 Heat + humidity can be draining — avoid heavy activity between 12-3 PM and drink water often.")
    if temp < 10:
        tips.append("🥶 Cold air can affect joints and breathing — layer up and keep extremities warm.")
    if rain_chance > 60:
        tips.append("🌧️ Roads may be slippery — walk and drive carefully.")
    if wind > 6:
        tips.append("💨 Strong winds today — secure loose items and be careful with umbrellas.")
    if humidity > 80:
        tips.append("😮‍💨 Very humid — those with asthma/allergies should keep medication handy.")

    if not tips:
        tips.append("✅ Conditions look moderate — no special precautions needed today.")

    return tips

# ---------------- Helper 4: Weather Summary ----------------
def get_summary(weather, temp, humidity, wind, rain_chance):
    emoji = WEATHER_EMOJI.get(weather, "🌈")
    return (
        f"{emoji} It looks like a **{weather}** day with a temperature of **{temp}°C**, "
        f"**{humidity}% humidity**, wind at **{wind} km/h**, and a **{rain_chance}% chance of rain**."
    )

# ---------------- Helper 5: Background color per weather ----------------
BACKGROUND_COLORS = {
    "sun": "#87CEEB",      # sky blue
    "rain": "#1B2A4A",     # dark blue
    "drizzle": "#5C7A99",  # muted blue-grey
    "snow": "#E8F0F5",     # icy white-blue
    "fog": "#8A8F94",      # grey
}

def apply_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
            transition: background-color 0.5s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- UI ----------------
st.title("🌦️ Smart Weather App")
st.write("Enter today's conditions and get a full weather-based forecast — from what to wear to how to stay healthy.")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("🌡️ Current Temperature (°C)", -10.0, 45.0, 20.0, 0.5)
    wind_speed = st.slider("💨 Wind Speed (km/h)", 0.0, 40.0, 10.0, 0.5)
with col2:
    humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
    rain_chance = st.slider("🌧️ Chance of Rain (%)", 0, 100, 20)

if st.button("🔍 Predict Weather", type="primary"):
    # The model was trained using the dataset's wind scale (0-9.5), so we scale
    # the km/h input down to roughly match that range for a sensible prediction.
    wind_for_model = wind_speed / 4

    input_df = pd.DataFrame([[temperature, wind_for_model]], columns=["temp_max", "wind"])
    predicted_weather = model.predict(input_df)[0]

    # Change the app's background color to match the predicted weather
    apply_background(BACKGROUND_COLORS.get(predicted_weather, "#0E1117"))

    st.divider()

    # 1. Weather
    emoji = WEATHER_EMOJI.get(predicted_weather, "🌈")
    st.subheader(f"{emoji} Weather: {predicted_weather.capitalize()}")

    # 2. Temperature Level
    st.subheader(f"🌡️ Temperature Level: {get_temp_level(temperature)}")

    # 3. What to Carry
    st.subheader("🎒 What to Carry")
    for item in get_carry_list(predicted_weather, humidity, rain_chance):
        st.write(f"- {item}")

    # 4. Health Recommendations
    st.subheader("🩺 Health Recommendations")
    for tip in get_health_tips(temperature, humidity, wind_speed, rain_chance):
        st.write(f"- {tip}")

    # 5. Weather Summary
    st.subheader("📋 Weather Summary")
    st.info(get_summary(predicted_weather, temperature, humidity, wind_speed, rain_chance))
