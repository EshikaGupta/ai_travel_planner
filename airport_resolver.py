"""
Airport / city / country -> IATA airport resolver.

Uses the public OurAirports CSV files as a local cache.
The files are downloaded only when they do not exist locally.

This resolver deliberately does NOT ask an LLM to invent IATA codes.
It resolves names against real airport records and can return multiple
airports when a location (city/country) has more than one airport.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


OURAIRPORTS_BASE = "https://ourairports.com/data/"
AIRPORTS_URL = OURAIRPORTS_BASE + "airports.csv"
COUNTRIES_URL = OURAIRPORTS_BASE + "countries.csv"

DATA_DIR = Path(__file__).resolve().parent / "data"
AIRPORTS_FILE = DATA_DIR / "airports.csv"
COUNTRIES_FILE = DATA_DIR / "countries.csv"


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    municipality: str
    country_code: str
    country_name: str
    airport_type: str
    latitude: str = ""
    longitude: str = ""


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AITravelPlanner/1.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())


def _load_country_names() -> dict[str, str]:
    _download_if_missing(COUNTRIES_URL, COUNTRIES_FILE)

    countries: dict[str, str] = {}

    with COUNTRIES_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            code = (row.get("code") or "").strip().upper()
            name = (row.get("name") or "").strip()

            if code and name:
                countries[code] = name

    return countries


def _load_airports() -> list[Airport]:
    _download_if_missing(AIRPORTS_URL, AIRPORTS_FILE)

    countries = _load_country_names()
    airports: list[Airport] = []

    with AIRPORTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            iata = (row.get("iata_code") or "").strip().upper()

            # Google Flights MCP needs IATA airport codes.
            if not iata or len(iata) != 3:
                continue

            airport_type = (row.get("type") or "").strip().lower()

            # Ignore small/heliport/seaplane/etc. airports for normal
            # commercial flight search.
            if airport_type not in {"large_airport", "medium_airport"}:
                continue

            country_code = (
                row.get("iso_country") or ""
            ).strip().upper()

            airports.append(
                Airport(
                    iata=iata,
                    name=(row.get("name") or "").strip(),
                    municipality=(row.get("municipality") or "").strip(),
                    country_code=country_code,
                    country_name=countries.get(country_code, ""),
                    airport_type=airport_type,
                    latitude=(row.get("latitude_deg") or "").strip(),
                    longitude=(row.get("longitude_deg") or "").strip(),
                )
            )

    return airports


def _normalise(value: str) -> str:
    value = value.strip().lower()

    replacements = {
        "new delhi": "delhi",
        "bombay": "mumbai",
        "calcutta": "kolkata",
        "madras": "chennai",
        "bengaluru": "bangalore",
    }

    value = replacements.get(value, value)

    # Makes "Tokyo, Japan" and similar inputs easier to match.
    value = re.sub(r"\s+", " ", value)

    return value


def _score_airport(
    airport: Airport,
    query: str,
) -> int:
    q = _normalise(query)

    score = 0

    if q == airport.iata.lower():
        score += 1000

    if q == _normalise(airport.municipality):
        score += 500

    if q == _normalise(airport.name):
        score += 400

    if q in _normalise(airport.name):
        score += 100

    if q in _normalise(airport.municipality):
        score += 150

    if airport.airport_type == "large_airport":
        score += 30

    return score


def resolve_airports(
    location: str,
    max_results: int = 5,
) -> list[Airport]:
    """
    Resolve a city, airport name, country, or IATA code to airports.

    Examples:
        Delhi -> [DEL]
        Tokyo -> [HND, NRT]
        Japan -> [HND, NRT, KIX, ...]
        DEL -> [DEL]

    Returns an empty list when no reliable match is found.
    """
    location = location.strip()

    if not location:
        return []

    airports = _load_airports()
    q = _normalise(location)

    # 1. Exact IATA match.
    if len(q) == 3 and q.isalpha():
        exact_iata = [
            airport
            for airport in airports
            if airport.iata == q.upper()
        ]

        if exact_iata:
            return exact_iata[:max_results]

    # 2. Strip a simple "City, Country" suffix.
    city_part = q.split(",", 1)[0].strip()

    # 3. Exact municipality match.
    city_matches = [
        airport
        for airport in airports
        if _normalise(airport.municipality) in {q, city_part}
    ]

    if city_matches:
        city_matches.sort(
            key=lambda airport: (
                0 if airport.airport_type == "large_airport" else 1,
                -_score_airport(airport, location),
            )
        )
        return city_matches[:max_results]

    # 4. Exact airport-name match.
    name_matches = [
        airport
        for airport in airports
        if _normalise(airport.name) == q
    ]

    if name_matches:
        return name_matches[:max_results]

    # 5. Country match.
    country_matches = [
        airport
        for airport in airports
        if _normalise(airport.country_name) == q
    ]

    if country_matches:
        # For country searches, prefer major airports.
        country_matches.sort(
            key=lambda airport: (
                0 if airport.airport_type == "large_airport" else 1,
                _normalise(airport.name),
            )
        )
        return country_matches[:max_results]

    # 6. Fuzzy-ish substring match for airport/city names.
    candidates = [
        airport
        for airport in airports
        if q in _normalise(airport.municipality)
        or q in _normalise(airport.name)
    ]

    candidates.sort(
        key=lambda airport: -_score_airport(airport, location)
    )

    return candidates[:max_results]


def describe_resolution(
    location: str,
    max_results: int = 5,
) -> str:
    """Return a human-readable resolution summary."""
    matches = resolve_airports(location, max_results=max_results)

    if not matches:
        return f"No airport could be resolved for '{location}'."

    lines = [f"'{location}' resolved to:"]

    for airport in matches:
        lines.append(
            f"- {airport.iata}: {airport.name}"
            f" ({airport.municipality}, {airport.country_name})"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    location = " ".join(sys.argv[1:]).strip()

    if not location:
        print("Usage: python airport_resolver.py Delhi")
        raise SystemExit(1)

    print(describe_resolution(location))
