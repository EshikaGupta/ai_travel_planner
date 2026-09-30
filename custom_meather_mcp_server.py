from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Weather Server")


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def _get(url, params):
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not configured."}
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if response.status_code != 200:
            return {"error": data.get("message", "Weather API request failed."), "status": response.status_code}
        return data
    except requests.RequestException as exc:
        return {"error": f"Weather API request failed: {exc}"}


@mcp.tool()
def get_current_weather(city: str):
    data = _get(
        "https://api.openweathermap.org/data/2.5/weather",
        {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
    )
    if "error" in data:
        return data

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }



@mcp.tool()
def get_forecast(city: str):
    data = _get(
        "https://api.openweathermap.org/data/2.5/forecast",
        {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
    )
    if "error" in data:
        return data

    forecast = [
        {
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "weather": item["weather"][0]["description"],
        }
        for item in data.get("list", [])[:5]
    ]

    return {"city": city, "forecast": forecast}


if __name__ == "__main__":
    mcp.run()