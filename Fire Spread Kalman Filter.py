#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FERDA v2.0 — Fire Emergency Response Drone Analysis                        ║
║  With IndianaMap Agriculture & Terrain Data Integration                     ║
║                                                                              ║
║  IndianaMap REST Services  : https://www.indianamap.org                     ║
║  Land Cover (NLCD 2006)    : maps.indiana.edu/arcgis/rest/services/         ║
║                               Environment/Land_Cover_2006                   ║
║  Soils (SSURGO 2019)       : .../Environment/Soils_SSURGO_Soil_Survey      ║
║  Water Bodies (NHD)        : .../Hydrology/Water_Bodies_Lakes_LocalRes      ║
║                                                                              ║
║  Fire Physics   : Rothermel + Andrews (1986) elliptical spread model        ║
║  State Estimate : 5-state Extended Kalman Filter (cx,cy,a,b,θ)             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Use a cross-platform font stack so Windows doesn't warn about missing glyphs
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
import requests
import json
import csv
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

# -----------------------------------------------------------------------------
# NLCD FUEL MODEL TABLE
# Maps National Land Cover Database codes to Rothermel-style fire parameters.
# Critical for Indiana agricultural landscapes:
#   Code 81 = Hay/Pasture, 82 = Cultivated Crops (dominant in Indiana)
# -----------------------------------------------------------------------------
NLCD_FUEL_MODELS = {
    11: {"name": "Open Water",            "mult": 0.00, "color": "#4472C4", "risk": "none"},
    21: {"name": "Developed - Open",      "mult": 0.30, "color": "#D47070", "risk": "low"},
    22: {"name": "Developed - Low",       "mult": 0.20, "color": "#C04040", "risk": "low"},
    23: {"name": "Developed - Medium",    "mult": 0.10, "color": "#9E2020", "risk": "low"},
    24: {"name": "Developed - High",      "mult": 0.05, "color": "#7A0000", "risk": "low"},
    31: {"name": "Barren Land",           "mult": 0.10, "color": "#B2B2B2", "risk": "low"},
    41: {"name": "Deciduous Forest",      "mult": 1.20, "color": "#38A800", "risk": "medium"},
    42: {"name": "Evergreen Forest",      "mult": 1.80, "color": "#1A5200", "risk": "high"},
    43: {"name": "Mixed Forest",          "mult": 1.50, "color": "#70A800", "risk": "high"},
    52: {"name": "Shrub/Scrub",           "mult": 1.60, "color": "#CDAA66", "risk": "high"},
    71: {"name": "Grassland/Herbaceous",  "mult": 1.40, "color": "#E6E600", "risk": "high"},
    81: {"name": "Hay/Pasture",           "mult": 1.30, "color": "#D6C500", "risk": "medium"},  # ← Indiana Ag
    82: {"name": "Cultivated Crops",      "mult": 1.10, "color": "#A87000", "risk": "medium"},  # ← Indiana Ag
    90: {"name": "Woody Wetlands",        "mult": 0.40, "color": "#7BA7BC", "risk": "low"},
    95: {"name": "Herbaceous Wetlands",   "mult": 0.60, "color": "#A6CEE3", "risk": "low"},
}
DEFAULT_NLCD = 82  # Cultivated Crops — most common in Indiana

# OpenStreetMap landuse/natural tags -> NLCD equivalent codes
# Used by the Overpass API fallback.
OSM_TO_NLCD = {
    # agriculture
    "farmland": 82, "farmyard": 82, "orchard": 82, "vineyard": 82,
    "allotments": 82, "plant_nursery": 82,
    # hay / pasture / grass
    "meadow": 81, "grass": 71, "recreation_ground": 71,
    "village_green": 71, "cemetery": 71, "flowerbed": 71,
    # forest
    "forest": 41, "logging": 41, "wood": 41,
    # shrub
    "scrub": 52, "heath": 52, "grassland": 71,
    # developed — universities/parks treated as open space or maintained grass
    "university": 21, "college": 21, "school": 21, "hospital": 21,
    "park": 71, "pitch": 71, "golf_course": 71, "nature_reserve": 71,
    # developed — higher density
    "residential": 22, "commercial": 23, "industrial": 23, "retail": 22,
    "office": 23, "civic": 22, "government": 22,
    "construction": 31, "brownfield": 31, "quarry": 31, "landfill": 31,
    # water / wetland
    "wetland": 90, "marsh": 95, "water": 11, "reservoir": 11, "basin": 11,
    # barren
    "bare_rock": 31, "sand": 31, "beach": 31,
}

# -----------------------------------------------------------------------------
# CDL FUEL MODEL TABLE  (USDA CropScape / Cropland Data Layer codes)
# CropScape uses different codes from NLCD. Indiana top crops: corn(1),
# soybeans(5), winter wheat(24), alfalfa(36), hay(37).
# -----------------------------------------------------------------------------
CDL_FUEL_MODELS = {
    1:   {"name": "Corn",                  "mult": 0.80, "risk": "low"},     # standing corn = wet
    2:   {"name": "Cotton",                "mult": 1.00, "risk": "medium"},
    3:   {"name": "Rice",                  "mult": 0.20, "risk": "none"},    # flooded fields
    4:   {"name": "Sorghum",               "mult": 1.10, "risk": "medium"},
    5:   {"name": "Soybeans",              "mult": 0.90, "risk": "medium"},
    6:   {"name": "Sunflower",             "mult": 1.30, "risk": "medium"},
    21:  {"name": "Barley",                "mult": 1.40, "risk": "high"},
    22:  {"name": "Durum Wheat",           "mult": 1.40, "risk": "high"},
    23:  {"name": "Spring Wheat",          "mult": 1.40, "risk": "high"},
    24:  {"name": "Winter Wheat",          "mult": 1.50, "risk": "high"},    # dry stubble
    26:  {"name": "Dbl Crop WinWht/Soy",  "mult": 1.20, "risk": "medium"},
    27:  {"name": "Rye",                   "mult": 1.35, "risk": "medium"},
    28:  {"name": "Oats",                  "mult": 1.30, "risk": "medium"},
    36:  {"name": "Alfalfa",               "mult": 1.10, "risk": "medium"},
    37:  {"name": "Other Hay/Non-Alfalfa", "mult": 1.30, "risk": "medium"},  # ← Indiana Ag
    59:  {"name": "Sod/Grass Seed",        "mult": 1.20, "risk": "medium"},
    61:  {"name": "Fallow/Idle Cropland",  "mult": 1.60, "risk": "high"},    # dry residue
    111: {"name": "Open Water",            "mult": 0.00, "risk": "none"},
    121: {"name": "Developed/Open Space",  "mult": 0.30, "risk": "low"},
    122: {"name": "Developed/Low",         "mult": 0.20, "risk": "low"},
    123: {"name": "Developed/Medium",      "mult": 0.10, "risk": "low"},
    124: {"name": "Developed/High",        "mult": 0.05, "risk": "low"},
    131: {"name": "Barren",                "mult": 0.10, "risk": "low"},
    141: {"name": "Deciduous Forest",      "mult": 1.20, "risk": "medium"},
    142: {"name": "Evergreen Forest",      "mult": 1.80, "risk": "high"},
    143: {"name": "Mixed Forest",          "mult": 1.50, "risk": "high"},
    152: {"name": "Shrubland",             "mult": 1.60, "risk": "high"},
    176: {"name": "Grassland/Pasture",     "mult": 1.40, "risk": "high"},    # ← Indiana Ag
    190: {"name": "Woody Wetlands",        "mult": 0.40, "risk": "low"},
    195: {"name": "Herbaceous Wetlands",   "mult": 0.60, "risk": "low"},
}
DEFAULT_CDL = 5   # Soybeans — #1 Indiana crop by area (alternates with corn)


RISK_COLORS = {
    "none":     "#4472C4",
    "low":      "#00B050",
    "medium":   "#FFC000",
    "high":     "#FF4500",
    "critical": "#9B00FF",
}

# UTM Zone 16N — Indiana's native CRS (EPSG:26916)
# Used for converting WGS84 lat/lon before querying maps.indiana.edu
def latlon_to_utm16n(lat: float, lon: float):
    """Simple WGS84 → UTM Zone 16N approximation (sufficient for identify queries)."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = 2 * f - f ** 2
    n = f / (2 - f)
    lon0 = math.radians(-87.0)  # Zone 16 central meridian
    phi = math.radians(lat)
    lam = math.radians(lon) - lon0
    e_p2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = e_p2 * math.cos(phi) ** 2
    A_ = math.cos(phi) * lam
    M = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64) * phi
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32) * math.sin(2 * phi)
        + (15 * e2 ** 2 / 256) * math.sin(4 * phi)
    )
    x = k0 * N * (A_ + (1 - T + C) * A_ ** 3 / 6) + 500000
    y = k0 * (M + N * math.tan(phi) * (A_ ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A_ ** 4 / 24))
    if lat < 0:
        y += 10000000
    return x, y


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------
@dataclass
class TerrainProfile:
    """Encapsulates all terrain / fuel data fetched from IndianaMap."""
    nlcd_code:         int   = DEFAULT_NLCD
    land_cover_name:   str   = "Cultivated Crops"
    fuel_mult:         float = 1.10
    soil_name:         str   = "Silty Clay Loam (Indiana default)"
    drainage_class:    str   = "Well drained"
    soil_moisture:     float = 0.35   # 0 = bone dry, 1 = saturated
    hydric_rating:     str   = "N"    # Y = hydric (wetland), N = not hydric
    has_water_nearby:  bool  = False
    slope_deg:         float = 0.0    # terrain slope in degrees (from 3DEP)
    risk_level:        str   = "medium"
    weather:           "WeatherData" = field(default_factory=lambda: WeatherData())
    lat:               float = 40.4259
    lon:               float = -86.9081

    def effective_mult(self) -> float:
        """Soil moisture dampens spread; hydric soils add further penalty."""
        m = 1.0 - self.soil_moisture * 0.65
        if self.hydric_rating == "Y":
            m *= 0.70
        if self.has_water_nearby:
            m *= 0.80
        return self.fuel_mult * m


@dataclass
class DroneObservation:
    """One drone fire-perimeter detection."""
    x:          float
    y:          float
    radius:     float
    timestamp:  float = 0.0
    drone_id:   int   = 0
    confidence: float = 1.0   # 0–1; scales measurement noise


@dataclass
class WeatherData:
    """Live weather fetched from Open-Meteo (free, no API key)."""
    wind_speed:    float = 0.0    # m/s at 10 m
    wind_dir_met:  float = 0.0    # meteorological: direction wind comes FROM (0=N CW)
    wind_dir_math: float = 270.0  # math convention: direction fire propagates (0=E CCW)
    humidity:      float = 50.0   # relative humidity %
    temperature:   float = 20.0   # Celsius
    precipitation: float = 0.0    # mm/h current
    weather_code:  int   = 0      # WMO code
    description:   str   = "Clear"

    @property
    def humidity_factor(self) -> float:
        """Rothermel ROS correction for atmospheric dryness (Rothermel 1972 Table 5 proxy).
        Dry air (<25% RH) increases effective spread rate by up to 45%.
        Saturated air (>85% RH) cuts spread by 65% via surface fuel moisture.
        """
        rh = self.humidity
        if rh >= 85: return 0.35
        if rh >= 65: return 0.60
        if rh >= 45: return 0.85
        if rh >= 25: return 1.00  # baseline
        if rh >= 15: return 1.25
        return 1.45

    @property
    def precipitation_factor(self) -> float:
        """Active rain/snow suppresses fire spread through fuel wetting."""
        p = self.precipitation
        if p >= 4.0: return 0.05
        if p >= 1.0: return 0.20
        if p >= 0.2: return 0.45
        return 1.00


WMO_DESCRIPTIONS = {
    0:"Clear", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Icy fog", 51:"Light drizzle", 53:"Moderate drizzle",
    55:"Dense drizzle", 61:"Light rain", 63:"Moderate rain", 65:"Heavy rain",
    71:"Light snow", 73:"Moderate snow", 75:"Heavy snow",
    80:"Light showers", 81:"Moderate showers", 82:"Heavy showers",
    95:"Thunderstorm", 96:"Thunderstorm+hail", 99:"Severe thunderstorm",
}


@dataclass
class AlertEvent:
    time:    float
    level:   str    # "INFO" | "WARNING" | "CRITICAL"
    message: str


# -----------------------------------------------------------------------------
# INDIANAMAP CLIENT
# -----------------------------------------------------------------------------
class IndianaMapClient:
    """
    Fetches terrain / agriculture data from public APIs — no API keys required.

    WORKING (confirmed):
      * USDA SDM   — soils       SDMDataAccess.nrcs.usda.gov
      * USGS NHD   — water       hydro.nationalmap.gov
      * OSM Overpass — land cover  overpass-api.de / kumi.systems mirrors

    REPLACED (were blocked/timing out):
      Removed: USGS MRLC, ESRI Living Atlas, USGS EPQS, USGS 3DEP (all blocked/offline)
    """

    # Land cover: OpenStreetMap Overpass (confirmed working)
    OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
    OVERPASS_MIRROR= "https://overpass.kumi.systems/api/interpreter"

    # Land cover backup: OSM Nominatim reverse geocoding (same OSM data, different server)
    NOMINATIM_URL  = "https://nominatim.openstreetmap.org/reverse"

    # Soils: USDA SDM Tabular REST (confirmed working)
    SDM_URL        = "https://SDMDataAccess.nrcs.usda.gov/tabular/post.rest"

    # Water bodies: USGS NHD (confirmed working)
    NHD_URL        = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"

    # Slope A: OpenTopoData — NED 10m for CONUS, no API key, different from USGS directly
    OPENTOPO_URL   = "https://api.opentopodata.org/v1/ned10m"

    # Slope B: open-elevation.com — SRTM 30m global, no API key
    OPENELEV_URL   = "https://api.open-elevation.com/api/v1/lookup"
    OPENMETEO_URL  = "https://api.open-meteo.com/v1/forecast"

    TIMEOUT = 8
    HEADERS = {"User-Agent": "FERDA-FireDetection/2.0 (drone fire spread tracker)"}

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cache: dict = {}
        self._cdl_failed = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"  |  {msg}")

    # ── 1. Land Cover ─────────────────────────────────────────────────────────
    def query_land_cover(self, lat: float, lon: float) -> int:
        """
        Source A: OSM Overpass is_in() query — confirmed working.
          Fetches all OSM area polygons containing the point, checks
          landuse / natural / leisure / amenity tags.

        Source B: OSM Nominatim reverse geocoding — same OSM data,
          completely different server (nominatim.openstreetmap.org).
          Returns the OSM object at the coordinate; we parse its tags.

        Both sources use the same OSM_TO_NLCD lookup table.
        """
        cache_key = f"nlcd_{lat:.4f}_{lon:.4f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # -- Source A: Overpass is_in -----------------------------------------
        query = (
            f"[out:json][timeout:8];"
            f"is_in({lat},{lon})->.a;"
            f"(area.a[landuse];area.a[natural];area.a[leisure];area.a[amenity];);"
            f"out tags;"
        )
        for mirror in (self.OVERPASS_URL, self.OVERPASS_MIRROR):
            self._log(f"Querying land cover [A] Overpass ({mirror.split('/')[2]})...")
            try:
                r = requests.post(mirror, data={"data": query},
                                  headers=self.HEADERS, timeout=self.TIMEOUT)
                if r.status_code == 200 and "html" not in r.headers.get("Content-Type","").lower():
                    for el in reversed(r.json().get("elements", [])):
                        tags = el.get("tags", {})
                        for key in ("landuse","natural","leisure","amenity"):
                            val = tags.get(key,"").lower()
                            if val in OSM_TO_NLCD:
                                code = OSM_TO_NLCD[val]
                                self._log(f"OSM {key}={val} -> NLCD {code}: {NLCD_FUEL_MODELS[code]['name']}")
                                self._cache[cache_key] = code
                                return code
                    self._log("Overpass: no matching tag — trying Nominatim")
                    break
                self._log(f"Overpass HTTP {r.status_code} — trying mirror")
            except Exception as e:
                self._log(f"Overpass failed ({type(e).__name__}) — trying mirror")

        # -- Source B: Nominatim reverse geocoding ----------------------------
        self._log("Querying land cover [B] OSM Nominatim reverse geocoding...")
        try:
            r = requests.get(
                self.NOMINATIM_URL,
                params={"lat": lat, "lon": lon, "format": "jsonv2",
                        "addressdetails": "0", "extratags": "1"},
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
            )
            data = r.json()
            # Nominatim returns extratags with landuse, natural etc.
            extra = data.get("extratags") or {}
            tags  = {**extra, "class": data.get("class",""), "type": data.get("type","")}
            for key in ("landuse","natural","leisure","amenity","class","type"):
                val = str(tags.get(key,"")).lower()
                if val in OSM_TO_NLCD:
                    code = OSM_TO_NLCD[val]
                    self._log(f"Nominatim {key}={val} -> NLCD {code}: {NLCD_FUEL_MODELS[code]['name']}")
                    self._cache[cache_key] = code
                    return code
        except Exception as e:
            self._log(f"Nominatim failed ({type(e).__name__})")

        self._cdl_failed = True
        self._log("Both land cover sources failed — will prompt manually")
        return DEFAULT_NLCD

    # ── 2. Soils (confirmed working — single source, no backup needed) ────────
    def query_soils(self, lat: float, lon: float) -> dict:
        """USDA SSURGO via SDM Tabular REST — spatial SQL for dominant component."""
        self._log("Querying soils (USDA SSURGO / SDM)...")
        defaults = {"name": "Silty Clay Loam (Indiana default)",
                    "drainage": "Well drained", "hydric": "N", "moisture": 0.35}
        sql = (
            f"SELECT TOP 1 co.compname, co.drainagecl, co.hydricrating "
            f"FROM mapunit mu INNER JOIN component co ON mu.mukey=co.mukey "
            f"INNER JOIN mupolygon mp ON mu.mukey=mp.mukey "
            f"WHERE co.majcompflag='Yes' "
            f"AND mp.mupolygongeo.STContains("
            f"geometry::STGeomFromText('POINT({lon} {lat})',4326))=1 "
            f"ORDER BY co.comppct_r DESC"
        )
        try:
            r = requests.post(self.SDM_URL,
                              data={"format":"JSON+COLUMNNAME","query":sql},
                              timeout=self.TIMEOUT)
            rows = r.json().get("Table", [])
            if len(rows) >= 2:
                row = dict(zip(rows[0], rows[1]))
                name   = str(row.get("compname", defaults["name"]))
                drain  = str(row.get("drainagecl", defaults["drainage"]))
                hydric = "Y" if str(row.get("hydricrating","No")).lower() in ("yes","y") else "N"
                moisture = {"Excessively drained":0.05,"Somewhat excessively drained":0.12,
                            "Well drained":0.25,"Moderately well drained":0.40,
                            "Somewhat poorly drained":0.55,"Poorly drained":0.70,
                            "Very poorly drained":0.85}.get(drain, 0.35)
                self._log(f"Soil: {name} | {drain} | Hydric: {hydric} | Moisture: {moisture:.0%}")
                return {"name":name,"drainage":drain,"hydric":hydric,"moisture":moisture}
        except Exception as e:
            self._log(f"SDM failed ({type(e).__name__})")
        return defaults

    # ── 3. Water bodies (confirmed working — single source, no backup needed) ──
    def query_water_nearby(self, lat: float, lon: float, buffer_deg: float = 0.005) -> bool:
        """USGS NHD REST — water body count within ~500m."""
        self._log("Checking water bodies (USGS NHD)...")
        xmin,xmax = lon-buffer_deg, lon+buffer_deg
        ymin,ymax = lat-buffer_deg, lat+buffer_deg
        try:
            r = requests.get(f"{self.NHD_URL}/0/query",
                params={"geometry":f"{xmin},{ymin},{xmax},{ymax}",
                        "geometryType":"esriGeometryEnvelope","inSR":"4326",
                        "spatialRel":"esriSpatialRelIntersects",
                        "returnCountOnly":"true","f":"json"},
                timeout=self.TIMEOUT)
            count = r.json().get("count", 0)
            found = count > 0
            self._log(f"Water bodies: {'YES' if found else 'No'}")
            return found
        except Exception as e:
            self._log(f"NHD failed ({type(e).__name__})")
            return False

    # ── 4. Slope — two independent sources, neither USGS ─────────────────────
    def query_slope(self, lat: float, lon: float) -> float:
        """
        Source A: OpenTopoData NED10m — wraps the same USGS NED 10m DEM but
          served from a non-USGS server (api.opentopodata.org). Accepts up to
          100 locations per request; we send 3 to compute finite-difference slope.

        Source B: open-elevation.com — SRTM 30m global coverage. Different
          network path, different data source (NASA SRTM vs USGS NED).

        Both use the same three-point finite difference:
          slope = arctan(max(|elev_N - elev_C|, |elev_E - elev_C|) / 100m)
        """
        dlat = 0.0009                                    # ~100 m in latitude
        dlon = 0.0009 / math.cos(math.radians(lat))     # ~100 m in longitude

        def rise_to_slope(c, n, e):
            return math.degrees(math.atan(max(abs(n-c), abs(e-c)) / 100.0))

        # -- Source A: OpenTopoData NED10m ------------------------------------
        self._log("Querying slope [A] OpenTopoData NED10m...")
        try:
            locs = (f"{lat},{lon}|{lat+dlat},{lon}|{lat},{lon+dlon}")
            r = requests.get(self.OPENTOPO_URL,
                             params={"locations": locs},
                             headers=self.HEADERS, timeout=self.TIMEOUT)
            results = r.json().get("results", [])
            if len(results) == 3:
                elev = [res["elevation"] for res in results]
                if None not in elev:
                    slope = rise_to_slope(*elev)
                    self._log(f"Slope: {slope:.2f} deg (OpenTopoData NED10m)")
                    return slope
        except Exception as e:
            self._log(f"OpenTopoData failed ({type(e).__name__})")

        # -- Source B: open-elevation.com SRTM --------------------------------
        self._log("Querying slope [B] open-elevation.com SRTM...")
        try:
            r = requests.post(self.OPENELEV_URL,
                json={"locations":[
                    {"latitude":lat,       "longitude":lon      },
                    {"latitude":lat+dlat,  "longitude":lon      },
                    {"latitude":lat,       "longitude":lon+dlon },
                ]},
                headers=self.HEADERS, timeout=self.TIMEOUT)
            results = r.json().get("results", [])
            if len(results) == 3:
                elev = [res["elevation"] for res in results]
                if None not in elev:
                    slope = rise_to_slope(*elev)
                    self._log(f"Slope: {slope:.2f} deg (open-elevation SRTM)")
                    return slope
        except Exception as e:
            self._log(f"open-elevation failed ({type(e).__name__})")

        self._log("Slope: both sources failed -- defaulting 0.0 deg (override in env prompt)")
        return 0.0

    def query_weather(self, lat: float, lon: float) -> "WeatherData":
        """
        Open-Meteo live weather — free, no API key, global coverage.
        Returns current wind speed/direction, relative humidity, temperature,
        precipitation, and WMO weather code.

        Wind direction conversion:
          Meteorological: direction wind comes FROM (0=north, 90=east, clockwise)
          Math/fire:      direction fire propagates (0=east, CCW)
          Formula: math_dir = (270 - met_dir) % 360
        """
        self._log("Querying live weather (Open-Meteo)...")
        try:
            r = requests.get(self.OPENMETEO_URL, params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "wind_speed_unit": "ms",
                "current": ("wind_speed_10m,wind_direction_10m,"
                            "relative_humidity_2m,temperature_2m,"
                            "precipitation,weather_code"),
            }, timeout=self.TIMEOUT)
            cur  = r.json().get("current", {})
            ws   = float(cur.get("wind_speed_10m",        0.0))
            wdm  = float(cur.get("wind_direction_10m",    0.0))
            rh   = float(cur.get("relative_humidity_2m",  50.0))
            temp = float(cur.get("temperature_2m",         20.0))
            prec = float(cur.get("precipitation",           0.0))
            wco  = int(cur.get("weather_code",               0))
            desc = WMO_DESCRIPTIONS.get(wco, f"WMO {wco}")
            math_dir = (270.0 - wdm) % 360.0
            self._log(
                f"Wind: {ws:.1f} m/s from {wdm:.0f}deg "
                f"-> propagates {math_dir:.0f}deg | "
                f"RH:{rh:.0f}% | {temp:.1f}C | Precip:{prec:.1f}mm/h | {desc}"
            )
            return WeatherData(
                wind_speed=ws, wind_dir_met=wdm, wind_dir_math=math_dir,
                humidity=rh, temperature=temp, precipitation=prec,
                weather_code=wco, description=desc,
            )
        except Exception as e:
            self._log(f"Open-Meteo failed ({type(e).__name__}) -- enter wind manually")
            return WeatherData()

    def build_terrain_profile(self, lat: float, lon: float) -> TerrainProfile:
        """
        Queries OSM Overpass/Nominatim (land cover), USDA SDM (soils), USGS NHD (water), OpenTopoData/open-elevation (slope), Open-Meteo (weather).
        If any service is unreachable, drops into an interactive picker
        so the operator always gets a meaningful terrain profile.
        """
        print(f"\n  +--- Terrain Fetch (USDA/USGS APIs) --- ({lat:.4f}N, {abs(lon):.4f}W) --")

        nlcd    = self.query_land_cover(lat, lon)
        soil    = self.query_soils(lat, lon)
        water   = self.query_water_nearby(lat, lon)
        slope   = self.query_slope(lat, lon)
        weather = self.query_weather(lat, lon)

        # ── Manual override when APIs were unreachable ─────────────────────
        if self._cdl_failed:
            print("  [!] Land cover API offline — please select terrain manually:")
            nlcd = self._pick_land_cover()


        fuel_data = NLCD_FUEL_MODELS.get(nlcd, NLCD_FUEL_MODELS[DEFAULT_NLCD])

        profile = TerrainProfile(
            nlcd_code       = nlcd,
            land_cover_name = fuel_data["name"],
            fuel_mult       = fuel_data["mult"],
            soil_name       = soil["name"],
            drainage_class  = soil["drainage"],
            soil_moisture   = soil["moisture"],
            hydric_rating   = soil["hydric"],
            has_water_nearby= water,
            slope_deg       = slope,
            risk_level      = fuel_data["risk"],
            weather         = weather,
            lat=lat, lon=lon,
        )
        print(f"  |")
        print(f"  |  * Land Cover  : {profile.land_cover_name}  (NLCD {nlcd})")
        print(f"  |  * Base Mult    : {profile.fuel_mult:.2f}  ->  Effective: {profile.effective_mult():.2f}")
        print(f"  |  * Soil         : {profile.soil_name[:40]}")
        print(f"  |  * Drainage     : {profile.drainage_class}  (moisture {profile.soil_moisture:.0%})")
        print(f"  |  * Hydric Soil  : {profile.hydric_rating}  (wetland penalty if Y)")
        print(f"  |  * Slope        : {slope:.1f} degrees")
        print(f"  |  * Weather      : {weather.description} | Wind {weather.wind_speed:.1f} m/s | RH {weather.humidity:.0f}% | Precip {weather.precipitation:.1f}mm/h")
        print(f"  |  * Water Nearby : {'YES -- spread penalty applied' if water else 'No'}")
        print(f"  |  * Risk Level   : {profile.risk_level.upper()}")
        print(f"  +-------------------------------------------------------------------\n")
        return profile



# Fuel load (kg/m2) by NLCD class — Scott & Burgan (2005) RMRS-GTR-153
# Used in Byram (1959) fireline intensity: I = H x w x r
NLCD_FUEL_LOAD = {
    11: 0.00,  42: 1.20,  71: 0.35,
    21: 0.10,  43: 1.00,  81: 0.40,
    22: 0.08,  52: 0.70,  82: 0.30,
    23: 0.05,  41: 0.80,  90: 0.20,
    24: 0.02,  31: 0.05,  95: 0.25,
}
DEFAULT_FUEL_LOAD = 0.30  # kg/m2 fallback


# -----------------------------------------------------------------------------
# 5-STATE EXTENDED KALMAN FILTER  [cx, cy, a, b, θ]
#   cx, cy  — fire centroid (m)
#   a       — semi-major axis (head-fire direction, m)
#   b       — semi-minor axis (flanking fire, m)
#   θ       — ellipse orientation (radians, = wind direction)
# -----------------------------------------------------------------------------
class FireKalmanFilter:
    # NIS chi-squared 95th percentile for dof=3 measurements
    NIS_95     = 7.815
    NIS_WINDOW = 8   # rolling window length

    def __init__(self, cx: float, cy: float, r: float):
        self.x = np.array([[cx], [cy], [r], [r * 0.65], [0.0]], dtype=float)
        self.P = np.diag([8.0, 8.0, 4.0, 4.0, 0.05])
        self.Q = np.diag([0.08, 0.08, 0.25, 0.18, 0.008])
        self.R_base = np.diag([4.0, 4.0, 7.0])
        self._nis_history: List[float] = []   # rolling NIS values

    # -- accessors ----------------------------------------------------------
    @property
    def cx(self): return float(self.x[0, 0])
    @property
    def cy(self): return float(self.x[1, 0])
    @property
    def semi_major(self): return max(float(self.x[2, 0]), 0.5)
    @property
    def semi_minor(self): return max(float(self.x[3, 0]), 0.5)
    @property
    def orientation_rad(self): return float(self.x[4, 0])
    @property
    def mean_radius(self): return (self.semi_major + self.semi_minor) / 2.0
    @property
    def area_m2(self): return math.pi * self.semi_major * self.semi_minor

    @property
    def confidence_pct(self) -> float:
        """
        Two-component confidence score:

        1. State uncertainty — how tight is P?
           Small trace(P) means the filter is certain about its estimate.
           (This alone caused the always-100% bug: every KF update shrinks P.)

        2. Innovation consistency (NIS) — are recent observations plausible
           given the model?
           NIS = v^T S^-1 v  (Normalised Innovation Squared)
           For a well-calibrated filter NIS ~ chi2(dof=3), 95th pct ~= 7.8.
           NIS >> 7.8 means observations are wildly inconsistent → confidence
           drops toward zero via an exponential penalty.

        A single wildly unrealistic observation drives NIS >> 100 and
        collapses confidence immediately. It recovers as the rolling window
        fills with consistent readings again.
        """
        # Component 1: state uncertainty (0–100)
        unc_score = min(35.0 / (np.trace(self.P) + 0.1) * 100.0, 100.0)

        if not self._nis_history:
            # No observations yet — return moderate uncertainty-based score
            return max(0.0, unc_score)

        # Component 2: innovation consistency penalty
        mean_nis    = min(float(np.mean(self._nis_history)), 500.0)
        # exp decay: no penalty at NIS~0; near-zero at NIS = 3 x NIS_95 (~23)
        consistency = math.exp(-mean_nis / (self.NIS_95 * 3.0))
        return max(0.0, min(100.0, unc_score * consistency))

    @property
    def last_nis(self) -> Optional[float]:
        """Latest NIS value, or None if no observations yet."""
        return self._nis_history[-1] if self._nis_history else None

    # -- Kalman predict step -------------------------------------------------
    def predict(self, dx: float, dy: float, da: float, db: float, theta: float):
        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0]  = max(self.x[2, 0] + da, 0.5)
        self.x[3, 0]  = max(self.x[3, 0] + db, 0.5)
        self.x[4, 0]  = theta
        self.P        = self.P + self.Q

    # -- Kalman update step --------------------------------------------------
    def update(self, obs: DroneObservation):
        """
        Standard KF update with pre-fit NIS computation.

        The innovation v = z - H*x is computed BEFORE the state update so
        it reflects how surprising the observation truly was. NIS is appended
        to the rolling window that drives confidence_pct.
        """
        H = np.array([
            [1, 0, 0.0, 0.0, 0],
            [0, 1, 0.0, 0.0, 0],
            [0, 0, 0.5, 0.5, 0],   # r_obs ~= mean(a, b)
        ], dtype=float)

        z   = np.array([[obs.x], [obs.y], [obs.radius]])
        R   = self.R_base / max(obs.confidence, 0.01)
        S   = H @ self.P @ H.T + R
        K   = self.P @ H.T @ np.linalg.inv(S)

        # Pre-fit innovation — captures how far this obs is from the prediction
        innov = z - H @ self.x

        # NIS: large value = very surprising / inconsistent observation
        nis = float((innov.T @ np.linalg.inv(S) @ innov)[0, 0])
        self._nis_history.append(nis)
        if len(self._nis_history) > self.NIS_WINDOW:
            self._nis_history.pop(0)

        self.x = self.x + K @ innov
        self.P = (np.eye(5) - K @ H) @ self.P
        self.x[2, 0] = max(self.x[2, 0], 0.5)
        self.x[3, 0] = max(self.x[3, 0], 0.5)


# -----------------------------------------------------------------------------
# ROTHERMEL + ANDREWS FIRE SPREAD MODEL
# Computes elliptical spread deltas for the KF predict step.
# Ref: Andrews (1986) — length-to-breadth ratio from wind speed.
# -----------------------------------------------------------------------------
class FireSpreadModel:
    BASE_ROS = 0.14          # m/s baseline rate-of-spread (dry Indiana cropland)
    MAX_LB   = 8.0           # maximum length-to-breadth ratio

    def __init__(self, terrain: TerrainProfile):
        self.terrain = terrain

    def compute_increments(self, wind_speed: float, wind_dir_deg: float,
                            slope_deg: float, dt: float):
        """
        Returns (dx, dy, da, db, theta_rad):
          dx, dy  — centroid displacement (m)
          da      — semi-major growth (m)
          db      — semi-minor growth (m)
          theta   — fire orientation (radians)
        """
        # Rothermel wind + slope adjustment factors
        phi_w = 0.10 * wind_speed ** 2
        phi_s = 5.275 * math.tan(math.radians(slope_deg)) ** 2

        # Effective terrain multiplier (includes soil moisture, hydric, water)
        eff = self.terrain.effective_mult()

        # Weather correction factors from Open-Meteo data
        wx = getattr(self.terrain, "weather", None)
        wx_factor = (wx.humidity_factor * wx.precipitation_factor
                     if wx is not None else 1.0)
        ros = self.BASE_ROS * (1 + phi_w + phi_s) * eff * wx_factor  # m/s

        # Andrews (1986) length-to-breadth ratio from 10-m wind speed
        U = wind_speed
        if U > 0.5:
            lb = min(0.936 * math.exp(0.2566 * U)
                     + 0.461 * math.exp(-0.1548 * U) - 0.397,
                     self.MAX_LB)
            lb = max(lb, 1.0)
        else:
            lb = 1.0

        da = ros * dt                          # head-fire (semi-major) increment
        db = (ros / lb) * dt                   # flanking  (semi-minor) increment

        # Centroid drifts ~50 % of head-fire speed toward head (wind direction)
        theta = math.radians(wind_dir_deg)
        drift = 0.50 * ros * dt
        dx = drift * math.cos(theta)
        dy = drift * math.sin(theta)

        return dx, dy, da, db, theta


# -----------------------------------------------------------------------------
# RISK ASSESSOR
# -----------------------------------------------------------------------------
class RiskAssessor:
    def __init__(self):
        self.alerts: List[AlertEvent] = []

    def assess(self, kf: FireKalmanFilter, terrain: TerrainProfile,
               wind_speed: float, t: float) -> str:
        area = kf.area_m2
        if area > 60_000 or (wind_speed > 12 and terrain.risk_level == "high"):
            lvl = "critical"
            self.alerts.append(AlertEvent(t, "CRITICAL",
                f"Fire area {area/10000:.1f} ha — immediate action required!"))
        elif area > 15_000 or terrain.risk_level == "high":
            lvl = "high"
            self.alerts.append(AlertEvent(t, "WARNING",
                f"High-risk terrain: {terrain.land_cover_name}"))
        elif area > 3_000:
            lvl = "medium"
        else:
            lvl = "low"
        return lvl


# -----------------------------------------------------------------------------
# FIRELINE CALCULATOR
#
# Byram, G.M. (1959) "Combustion of forest fuels." In: Davis KP (ed)
#   Forest Fire: Control and Use. McGraw-Hill. pp 61-89.
#   Fireline intensity: I = H x w x r  (kW/m)
#   Flame length:       L = 0.0775 x I^0.46  (m)
#
# NWCG PMS 410-1 (2004) Fireline Handbook, Appendix A (attack thresholds):
#   < 500 kW/m  : direct attack, hand tools feasible
#   < 2000 kW/m : mechanized (dozer/engine) attack feasible
#   < 4000 kW/m : aerial retardant may be effective
#   >= 4000 kW/m: indirect attack only — no safe direct approach
#
# Lindsey & Bratten (1983) / NWCG PMS 410-1:
#   Safety zone diameter >= 4 x flame_length
#
# Andrews (1986) BEHAVE INT-194:
#   Optimal fireline: flank of ellipse, anchored at back-of-fire,
#   offset laterally by semi-minor + 20% safety buffer.
# -----------------------------------------------------------------------------
class FirelineCalculator:
    H_COMBUSTION   = 18_000.0  # kJ/kg heat of combustion (Rothermel 1972 Table 1)
    THRESH_HAND    =    500.0  # kW/m  NWCG: hand crew direct attack feasible
    THRESH_DOZER   =  2_000.0  # kW/m  NWCG: dozer/engine feasible
    THRESH_AERIAL  =  4_000.0  # kW/m  NWCG: aerial retardant may work
    SAFETY_FACTOR  =      4.0  # safety zone diam = 4 x flame_length (Lindsey & Bratten 1983)

    def __init__(self, terrain: "TerrainProfile"):
        self.terrain = terrain

    def byram_intensity(self, ros_ms: float) -> float:
        """Byram (1959): I = H x w x r  [kW/m]"""
        w = NLCD_FUEL_LOAD.get(self.terrain.nlcd_code, DEFAULT_FUEL_LOAD)
        return self.H_COMBUSTION * w * ros_ms

    def flame_length(self, intensity_kw: float) -> float:
        """Byram (1959): L = 0.0775 x I^0.46  [m]"""
        return 0.0775 * (max(intensity_kw, 0.0) ** 0.46) if intensity_kw > 0 else 0.0

    def attack_method(self, intensity_kw: float):
        """NWCG PMS 410-1 attack feasibility by fireline intensity."""
        if intensity_kw < self.THRESH_HAND:
            return "DIRECT - hand tools", "#00FF7F"
        if intensity_kw < self.THRESH_DOZER:
            return "DIRECT - dozer/engine", "#FFD700"
        if intensity_kw < self.THRESH_AERIAL:
            return "AERIAL retardant + indirect", "#FFA500"
        return "INDIRECT ONLY - no safe direct attack", "#FF4444"

    def safety_zone_radius(self, flame_len_m: float) -> float:
        """Min safety zone radius: diam >= 4 x flame_length (Lindsey & Bratten 1983)."""
        return self.SAFETY_FACTOR * flame_len_m / 2.0

    def recommended_fireline(self, kf, wind_dir_rad: float) -> dict:
        """
        Andrews (1986) BEHAVE optimal fireline placement.
        Fireline runs along the LEFT FLANK of the fire ellipse, from an
        anchor at the back-of-fire to the head, offset by semi-minor + 20%.
        """
        cx, cy = kf.cx, kf.cy
        a, b   = kf.semi_major, kf.semi_minor
        ct, st = math.cos(wind_dir_rad), math.sin(wind_dir_rad)
        # perpendicular (left flank = 90 deg CCW from wind)
        px, py = -st, ct
        offset = b * 1.20  # semi-minor + 20% buffer
        back = (cx - a*ct + px*offset, cy - a*st + py*offset)
        tip  = (cx + a*ct + px*offset, cy + a*st + py*offset)
        return {"anchor": back, "tip": tip,
                "length": math.hypot(tip[0]-back[0], tip[1]-back[1])}



# -----------------------------------------------------------------------------
# FERDA v2.0 — MAIN SYSTEM
# -----------------------------------------------------------------------------
class FERDA_System:
    def __init__(self, ix: float, iy: float, ir: float, terrain: TerrainProfile):
        self.kf           = FireKalmanFilter(ix, iy, ir)
        self.spread       = FireSpreadModel(terrain)
        self.terrain      = terrain
        self.risk         = RiskAssessor()
        self.fireline     = FirelineCalculator(terrain)
        self._topo_grid   = None   # (X, Y, Z) numpy arrays fetched once
        self._topo_contour= None   # matplotlib QuadContourSet handle
        self._fl_line     = None   # fireline artist
        self._sz_circle   = None   # safety zone artist
        self.current_time = 0.0
        self.wind_speed   = 0.0
        self.wind_dir     = 0.0

        self.intensity_history: List[float] = [0.0]
        self.history: List[dict] = [{"t": 0, "a": ir, "b": ir * 0.65,
                                      "area": math.pi * ir * ir * 0.65}]
        self.observations: List[DroneObservation] = []
        self.log_records:  List[list] = []

        self._build_figure()

    # -- TOPO GRID -----------------------------------------------------------
    def _fetch_topo_grid(self, n: int = 16):
        """
        Queries open-elevation.com for an n x n elevation grid centred on
        the fire origin, spanning a SYMMETRIC square of
        +/- self._TERRAIN_HALF_SPAN metres around (0,0).
        Converts local (x, y) metres to WGS84 offsets, batches into one
        POST request, and returns (X, Y, Z) numpy arrays for contour plotting.
        """
        H = getattr(self, "_TERRAIN_HALF_SPAN", 500)
        xs = np.linspace(-H, H, n)
        ys = np.linspace(-H, H, n)
        lat0, lon0 = self.terrain.lat, self.terrain.lon
        # 1 degree lat ~= 111,320 m; 1 degree lon ~= 111,320 * cos(lat) m
        dlat_per_m = 1.0 / 111_320.0
        dlon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat0)))

        locations = []
        for y in ys:
            for x in xs:
                locations.append({
                    "latitude":  lat0 + y * dlat_per_m,
                    "longitude": lon0 + x * dlon_per_m,
                })
        try:
            r = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations},
                headers={"User-Agent": "FERDA-FireDetection/2.0"},
                timeout=20,
            )
            results = r.json().get("results", [])
            if len(results) == n * n:
                elevs = np.array([res["elevation"] for res in results],
                                 dtype=float).reshape(n, n)
                X, Y = np.meshgrid(xs, ys)
                print(f"  Topo grid: {n}x{n} pts, "
                      f"elev range {elevs.min():.0f}-{elevs.max():.0f} m ASL")
                return X, Y, elevs
        except Exception as e:
            print(f"  Topo grid fetch failed ({type(e).__name__}) — skipping contours")
        return None

    # -- FIGURE ---------------------------------------------------------------
    # ── PROCEDURAL TERRAIN ────────────────────────────────────────────────────
    # Below this total relief (m), real elevation data is mostly sensor noise/
    # interpolation artifacts rather than meaningful terrain — treat as flat.
    FLAT_RELIEF_THRESHOLD_M = 3.0

    def _make_terrain(self, span=None, n=200):
        H = getattr(self, "_TERRAIN_HALF_SPAN", 500)
        xs = np.linspace(-H, H, n)
        ys = np.linspace(-H, H, n)
        X, Y = np.meshgrid(xs, ys)
        topo = self._fetch_topo_grid(n=18)
        self.is_flat_terrain = False

        if topo is not None:
            try:
                from scipy.interpolate import RegularGridInterpolator
                Xc, Yc, Zc = topo
                rgi = RegularGridInterpolator(
                    (Xc[0, :], Yc[:, 0]), Zc.T,
                    method='linear', bounds_error=False, fill_value=None)
                Z = rgi((X, Y))
                # Real elevation data is coarse (often 30-90m source resolution)
                # interpolated onto a fine grid — this produces fake banding.
                # Gaussian-smooth to remove interpolation/quantization artifacts.
                try:
                    from scipy.ndimage import gaussian_filter
                    Z = gaussian_filter(Z, sigma=max(n / 25, 3))
                except ImportError:
                    pass
                relief = float(np.nanmax(Z) - np.nanmin(Z))
                if relief < self.FLAT_RELIEF_THRESHOLD_M:
                    # Relief this small is noise, not real terrain shape.
                    # Flatten completely rather than show misleading banding.
                    self.is_flat_terrain = True
                    Z = np.full_like(Z, np.nanmean(Z))
            except Exception:
                topo = None

        if topo is None:
            rng = np.random.default_rng(
                int(abs(self.terrain.lat * 1000) + abs(self.terrain.lon * 1000)))
            Z = np.zeros((n, n))
            for amp, freq in [(80, 0.006), (30, 0.015), (12, 0.04), (4, 0.10)]:
                ph = rng.uniform(0, 2 * np.pi, (2,))
                Z += amp * (np.sin(freq * X + ph[0]) * np.cos(freq * Y + ph[1]))
            Z -= Z.min()

        dz_dx = np.gradient(Z, axis=1)
        dz_dy = np.gradient(Z, axis=0)
        sun = np.array([-1.0, -1.0, 2.0])
        sun /= np.linalg.norm(sun)
        norm = np.stack([-dz_dx, -dz_dy, np.ones_like(Z)], axis=-1)
        nlen = np.linalg.norm(norm, axis=-1, keepdims=True)
        nlen[nlen == 0] = 1.0
        norm /= nlen
        shade = np.clip((norm * sun).sum(axis=-1), 0.0, 1.0)
        if self.is_flat_terrain:
            shade = np.full_like(shade, 0.75)   # flat, evenly lit
        return X, Y, Z, shade

    def _fuel_alpha_map(self, X, Y):
        rng = np.random.default_rng(42)
        base = self.terrain.effective_mult()
        noise = np.zeros_like(X)
        for amp, freq in [(0.3, 0.012), (0.15, 0.030)]:
            ph = rng.uniform(0, 2 * np.pi, (2,))
            noise += amp * np.sin(freq * X + ph[0]) * np.cos(freq * Y + ph[1])
        return np.clip(base * 0.5 + noise, 0.0, 1.0)

    # ── FIRELINE ADVISOR ──────────────────────────────────────────────────────
    def _advise_firelines(self):
        kf    = self.kf
        fc    = self.fireline
        theta = kf.orientation_rad
        ct, st = math.cos(theta), math.sin(theta)
        px, py = -st,  ct
        qx, qy =  st, -ct
        a, b   = kf.semi_major, kf.semi_minor
        cx, cy = kf.cx, kf.cy
        ros    = getattr(self, "_last_ros", 0.0)
        intensity = fc.byram_intensity(ros)

        off = b * 1.20
        candidates = []

        # 1: Left flank
        back1 = (cx - a*ct + px*off, cy - a*st + py*off)
        head1 = (cx + a*ct + px*off, cy + a*st + py*off)
        steep = self.terrain.slope_deg > 20
        r1 = ("Caution: steep slope slows crews" if steep
              else "Preferred: parallel to spread, anchored at back-of-fire")
        candidates.append(dict(
            anchor=back1, tip=head1,
            length=math.hypot(head1[0]-back1[0], head1[1]-back1[1]),
            color="#00E676", rank=1, label="FL-1  Left flank", reason=r1))

        # 2: Right flank
        back2 = (cx - a*ct + qx*off, cy - a*st + qy*off)
        head2 = (cx + a*ct + qx*off, cy + a*st + qy*off)
        low_fuel = self.terrain.effective_mult() < 0.8
        r2 = ("Low fuel load reduces fireline intensity" if low_fuel
              else "Secondary option if left flank inaccessible")
        candidates.append(dict(
            anchor=back2, tip=head2,
            length=math.hypot(head2[0]-back2[0], head2[1]-back2[1]),
            color="#FFD600", rank=2, label="FL-2  Right flank", reason=r2))

        # 3: Indirect intercept ahead of head fire
        adv = a * 2.0
        mid3  = (cx + ct*adv,        cy + st*adv)
        back3 = (mid3[0] + px*b*2,   mid3[1] + py*b*2)
        head3 = (mid3[0] + qx*b*2,   mid3[1] + qy*b*2)
        r3 = ("Required: intensity too high for direct attack — NWCG PMS 410-1"
              if intensity >= fc.THRESH_DOZER
              else "Use if flanks inaccessible; requires rapid deployment")
        candidates.append(dict(
            anchor=back3, tip=head3,
            length=math.hypot(head3[0]-back3[0], head3[1]-back3[1]),
            color="#FF6D00", rank=3, label="FL-3  Indirect intercept", reason=r3))

        return candidates

    # ── _BUILD_FIGURE ──────────────────────────────────────────────────────────
    def _build_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 11), facecolor="#0d0f1a")
        self.fig.subplots_adjust(left=0.03, right=0.72, top=0.95, bottom=0.04)
        # Terrain is rendered as a SYMMETRIC square centred on the fire
        # origin (0,0), not an offset [-80, 650] box -- that put the fire
        # near the corner instead of the middle, wasting most of the
        # available buffer before dynamic zoom hit the edge.
        self._TERRAIN_HALF_SPAN = 500
        self._SPAN = self._TERRAIN_HALF_SPAN   # kept for compatibility

        # Main map
        self.ax = self.fig.add_axes([0.03, 0.04, 0.66, 0.89])
        self.ax.set_facecolor("#0d0f1a")
        self.ax.set_aspect("equal")
        for sp in self.ax.spines.values():
            sp.set_color("#1e2240")
        self.ax.tick_params(colors="#3a3f6a", labelsize=7)
        self.ax.set_xlabel("X (m)", color="#3a3f6a", fontsize=8)
        self.ax.set_ylabel("Y (m)", color="#3a3f6a", fontsize=8)
        self.ax.set_xlim(-100, 100)   # placeholder; render() sets real zoom
        self.ax.set_ylim(-100, 100)

        # Terrain layers
        print("  Generating terrain layer...")
        X, Y, Z, shade = self._make_terrain(span=self._SPAN)
        self._terrain_xyz = (X, Y, Z)

        if getattr(self, "is_flat_terrain", False):
            # Real elevation data showed <3m relief -- that's noise, not
            # terrain. Show a flat neutral ground colour instead of fake
            # contours/banding, and say so plainly.
            self.ax.imshow(np.ones_like(Z) * 0.6,
                           extent=[-self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN,
                               -self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN],
                           origin="lower", cmap="gray", vmin=0, vmax=1,
                           alpha=0.30, zorder=1)
            self.ax.text(0.015, 0.985, "Terrain: flat (<3m relief)",
                         transform=self.ax.transAxes, fontsize=7,
                         color="#788", va="top", ha="left", zorder=20)
        else:
            self.ax.imshow(shade, extent=[-self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN,
                               -self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN],
                           origin="lower", cmap="gray", vmin=0, vmax=1,
                           alpha=0.55, zorder=1, interpolation="bilinear")
            im = self.ax.imshow(Z, extent=[-self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN,
                               -self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN],
                                origin="lower", cmap="YlOrBr", alpha=0.35,
                                zorder=2, interpolation="bilinear")
            cl = self.ax.contour(X, Y, Z, levels=5, colors="#ffffff",
                                 alpha=0.20, linewidths=0.6, zorder=3)
            self.ax.clabel(cl, cl.levels[::2], inline=True, fontsize=5,
                           fmt="%.0fm", colors="#999999")
            cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.025, pad=0.01)
            cbar.set_label("Elevation (m ASL)", color="#3a3f6a", fontsize=6.5)
            cbar.ax.tick_params(colors="#3a3f6a", labelsize=5.5)

        # Fuel density overlay
        fuel = self._fuel_alpha_map(X, Y)
        self.ax.imshow(fuel, extent=[-self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN,
                               -self._TERRAIN_HALF_SPAN, self._TERRAIN_HALF_SPAN],
                       origin="lower", cmap="Greens", alpha=0.22,
                       vmin=0, vmax=1, zorder=4, interpolation="bilinear")

        # Fire patches
        self._pred_patches = []
        self._patch_fire = Ellipse((0, 0), 1, 1, angle=0,
                                   facecolor="#FF3D00", edgecolor="#FFAB40",
                                   linewidth=1.8, alpha=0.85, zorder=8)
        self._patch_uncert = Ellipse((0, 0), 1, 1, angle=0,
                                     edgecolor="#FFD600", facecolor="none",
                                     linestyle="--", linewidth=1.2,
                                     alpha=0.45, zorder=9)
        self._patch_sz = Ellipse((0, 0), 1, 1, angle=0,
                                  edgecolor="#00B0FF", facecolor="#00B0FF",
                                  alpha=0.08, linewidth=1.4,
                                  linestyle=":", zorder=9)
        for p in (self._patch_fire, self._patch_uncert, self._patch_sz):
            self.ax.add_patch(p)
        # Ghost trail: capped rolling history, not infinite accumulation
        self._ghost_patches = []
        self._GHOST_MAX = 6
        # Spread-prediction text labels, cleared and redrawn each frame
        self._pred_labels = []

        self._quiver = self.ax.quiver(
            0, 0, 0, 0,
            color="#64B5F6", scale=200, width=0.005,
            headwidth=5, headlength=6, zorder=12)
        self._wind_label = self.ax.text(
            0, 0, "WIND", color="#64B5F6", fontsize=6.5,
            ha="center", va="bottom", zorder=12)

        self._fl_artists = []
        self._obs_artists = {}

        # Right panel — three stacked info boxes
        px0, pw = 0.73, 0.265

        hdr = self.fig.add_axes([px0, 0.93, pw, 0.05])
        hdr.axis("off")
        hdr.text(0.5, 0.5, "FERDA  Decision Support",
                 ha="center", va="center", fontsize=11,
                 fontweight="bold", color="#E8EAF6")

        def _panel(y, h, title, fontsize=8.5, linespacing=1.7):
            a = self.fig.add_axes([px0, y, pw, h])
            a.set_facecolor("#10122a")
            for sp in a.spines.values():
                sp.set_color("#1e2240")
            a.axis("off")
            a.text(0.04, 0.97, title, transform=a.transAxes,
                   fontsize=8, fontweight="bold", color="#90CAF9", va="top")
            return a.text(0.04, 0.87, "", transform=a.transAxes,
                          fontsize=fontsize, va="top", color="#E8EAF6",
                          fontfamily="monospace", linespacing=linespacing)

        # Fireline panel needs ~2x the vertical room (3 candidates x 3 lines
        # each) compared to status/spread, which are fixed at ~8 lines each.
        self._txt_status = _panel(0.70, 0.22, "FIRE STATUS",
                                   fontsize=8.0, linespacing=1.55)
        self._txt_spread = _panel(0.47, 0.21, "SPREAD FORECAST",
                                   fontsize=8.0, linespacing=1.55)
        self._txt_fireln = _panel(0.04, 0.41, "FIRELINE RECOMMENDATIONS",
                                   fontsize=7.3, linespacing=1.42)

    # ── RENDER ────────────────────────────────────────────────────────────────
    def render(self):
        kf    = self.kf
        theta = kf.orientation_rad
        ct, st = math.cos(theta), math.sin(theta)
        deg   = math.degrees(theta)

        # -- Dynamic zoom: keep the fire framed at a useful scale --------------
        # Track the largest extent the fire has reached (including the 6h
        # prediction horizon) so the view doesn't jitter as the fire shrinks
        # slightly between Kalman updates -- it only zooms OUT, never in,
        # below a sensible minimum.
        #
        # HARD CAP: the rendered terrain image only covers a fixed area
        # (self._TERRAIN_HALF_SPAN around the origin). If the computed zoom
        # radius exceeds that, capping it here prevents the camera from
        # showing empty background outside the terrain image -- which is
        # what produced the "dark void" bug. The 6h spread ellipse may still
        # extend past the terrain edge in extreme-wind scenarios; that's a
        # visual edge case, not a black hole in the map.
        dx6, dy6, da6, db6, _ = self.spread.compute_increments(
            self.wind_speed, self.wind_dir, self.terrain.slope_deg, 21600)
        reach = max(kf.semi_major + da6, kf.semi_minor + db6,
                    abs(dx6), abs(dy6), 1.0)
        raw_radius  = max(getattr(self, "_max_view_radius", 0.0), reach * 2.2, 40.0)
        view_radius = min(raw_radius, self._TERRAIN_HALF_SPAN)
        self._max_view_radius = view_radius
        cx, cy = kf.cx, kf.cy
        self.ax.set_xlim(cx - view_radius, cx + view_radius)
        self.ax.set_ylim(cy - view_radius, cy + view_radius)

        risk_lvl = self.risk.assess(
            kf, self.terrain, self.wind_speed, self.current_time)
        RCOL = {"none":"#4CAF50","low":"#4CAF50","medium":"#FFB300",
                "high":"#FF6D00","critical":"#D500F9"}
        fire_col = RCOL.get(risk_lvl, "#FF3D00")

        ros_now   = getattr(self, "_last_ros", 0.0)
        intensity = self.fireline.byram_intensity(ros_now)
        flame_len = self.fireline.flame_length(intensity)
        sz_rad    = self.fireline.safety_zone_radius(flame_len)
        atk_str, _ = self.fireline.attack_method(intensity)

        # Current fire ellipse — single clean outline, no glow stack
        self._patch_fire.center = (kf.cx, kf.cy)
        self._patch_fire.width  = kf.semi_major * 2
        self._patch_fire.height = kf.semi_minor * 2
        self._patch_fire.angle  = deg
        self._patch_fire.set_facecolor(fire_col)

        pos_std = math.sqrt(kf.P[0, 0] + kf.P[1, 1])
        self._patch_uncert.center = (kf.cx, kf.cy)
        self._patch_uncert.width  = kf.semi_major * 2 + pos_std * 2
        self._patch_uncert.height = kf.semi_minor * 2 + pos_std * 2
        self._patch_uncert.angle  = deg

        bx = kf.cx - kf.semi_major * ct
        by = kf.cy - kf.semi_major * st
        self._patch_sz.center = (bx, by)
        d = max(sz_rad * 2, 5)
        self._patch_sz.width = d; self._patch_sz.height = d

        # Ghost trail — capped rolling history (was unbounded, causing clutter)
        ghost = Ellipse((kf.cx, kf.cy), kf.semi_major*2, kf.semi_minor*2,
                        angle=deg, edgecolor="#FF8A50", facecolor="none",
                        alpha=0.10, linestyle=":", linewidth=0.8, zorder=6)
        self.ax.add_patch(ghost)
        self._ghost_patches.append(ghost)
        while len(self._ghost_patches) > self._GHOST_MAX:
            old = self._ghost_patches.pop(0)
            old.remove()

        # Predicted spread ellipses (1h / 3h / 6h) — clear previous frame fully
        for p in self._pred_patches:
            p.remove()
        self._pred_patches = []
        for lbl_txt in self._pred_labels:
            lbl_txt.remove()
        self._pred_labels = []

        for dt_sec, alpha, lbl in [(3600, 0.16, "1h"),
                                   (10800, 0.10, "3h"),
                                   (21600, 0.06, "6h")]:
            dx, dy, da, db, _ = self.spread.compute_increments(
                self.wind_speed, self.wind_dir,
                self.terrain.slope_deg, dt_sec)
            pa = max(kf.semi_major + da, 1)
            pb = max(kf.semi_minor + db, 1)
            ep = Ellipse((kf.cx + dx, kf.cy + dy), pa*2, pb*2,
                         angle=deg, facecolor=fire_col,
                         edgecolor=fire_col, linewidth=0.6,
                         alpha=alpha, zorder=5)
            self.ax.add_patch(ep)
            self._pred_patches.append(ep)
            lx = kf.cx + dx + pa * ct
            ly = kf.cy + dy + pa * st
            _xl, _yl = self.ax.get_xlim(), self.ax.get_ylim()
            if _xl[0] < lx < _xl[1] and _yl[0] < ly < _yl[1]:
                t = self.ax.text(lx, ly, lbl, fontsize=6.5, color="#FFCC02",
                                 ha="center", va="bottom", zorder=13,
                                 fontweight="bold")
                self._pred_labels.append(t)

        # Wind — reposition into the top-right corner of the CURRENT zoomed view
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        wx_pos = xlim[1] - (xlim[1]-xlim[0]) * 0.08
        wy_pos = ylim[1] - (ylim[1]-ylim[0]) * 0.08
        self._quiver.set_offsets([[wx_pos, wy_pos]])
        self._quiver.set_UVC(
            math.cos(math.radians(self.wind_dir)) * self.wind_speed,
            math.sin(math.radians(self.wind_dir)) * self.wind_speed)
        self._wind_label.set_position((wx_pos, wy_pos + (ylim[1]-ylim[0])*0.03))

        # Drone markers
        drone_cols = ["#FF80AB","#80D8FF","#CCFF90","#FFD180","#EA80FC"]
        for obs in self.observations[-30:]:
            did = obs.drone_id
            col = drone_cols[did % len(drone_cols)]
            if did not in self._obs_artists:
                ln, = self.ax.plot([], [], marker="x", markersize=9,
                                   linewidth=0, color=col, zorder=11,
                                   markeredgewidth=2)
                self._obs_artists[did] = ln
            pts = [(o.x, o.y) for o in self.observations if o.drone_id==did]
            if pts:
                self._obs_artists[did].set_data(*zip(*pts))

        # Firelines
        candidates = self._advise_firelines()
        for artist_set in self._fl_artists:
            for artist in artist_set:
                try: artist.remove()
                except Exception: pass
        self._fl_artists = []
        for c in candidates:
            ax0, ti = c["anchor"], c["tip"]
            ln, = self.ax.plot([ax0[0], ti[0]], [ax0[1], ti[1]],
                               color=c["color"], linewidth=2.2,
                               linestyle="-", zorder=10, alpha=0.85)
            mk, = self.ax.plot(ax0[0], ax0[1], marker="^",
                               color=c["color"], markersize=8,
                               zorder=11, linewidth=0)
            lbl = self.ax.text(ax0[0], ax0[1] - 14, str(c["rank"]),
                               color=c["color"], fontsize=7,
                               fontweight="bold", ha="center", va="top",
                               zorder=12)
            self._fl_artists.append((ln, mk, lbl))

        # Map title
        wx = getattr(self.terrain, "weather", None)
        wx_s = (f"  |  {wx.description}  {wx.wind_speed:.1f}m/s  RH{wx.humidity:.0f}%"
                if wx and wx.wind_speed > 0 else "")
        self.ax.set_title(
            f"{self.terrain.land_cover_name}"
            f"  ({self.terrain.lat:.4f}N, {abs(self.terrain.lon):.4f}W)"
            f"{wx_s}",
            color="#B0BEC5", fontsize=8.5, pad=6)

        # STATUS panel
        conf    = kf.confidence_pct
        nis_str = f"{kf.last_nis:.1f}" if kf.last_nis is not None else "n/a"
        conf_col = "#4CAF50" if conf > 70 else "#FFB300" if conf > 40 else "#FF5252"
        self._txt_status.set_text(
            f"Time   {self.current_time:>8.0f} s\n"
            f"Risk   {risk_lvl.upper()}\n"
            f"Conf   {conf:>7.1f}%  (NIS {nis_str})\n"
            f"Center ({kf.cx:.0f}, {kf.cy:.0f}) m\n"
            f"Axes   {kf.semi_major:.0f} x {kf.semi_minor:.0f} m\n"
            f"Area   {kf.area_m2/10_000:.3f} ha\n"
            f"Wind   {self.wind_speed:.1f} m/s @ {self.wind_dir:.0f} deg\n"
            f"Slope  {self.terrain.slope_deg:.1f} deg"
        )
        self._txt_status.set_color(conf_col)

        # SPREAD FORECAST panel
        rows = []
        for dt_sec, lbl in [(3600,"1 hr"), (10800,"3 hr"), (21600,"6 hr")]:
            _, _, da, db, _ = self.spread.compute_increments(
                self.wind_speed, self.wind_dir,
                self.terrain.slope_deg, dt_sec)
            pa = kf.semi_major + da
            pb = kf.semi_minor + db
            ha = math.pi * pa * pb / 10_000
            rows.append(f"{lbl}  {pa:.0f}x{pb:.0f}m  {ha:.2f}ha")
        self._txt_spread.set_text(
            "Horizon  Axes          Area\n"
            + "\n".join(rows)
            + f"\n\nByram I  {intensity:.0f} kW/m\n"
            + f"Flame    {flame_len:.1f} m\n"
            + f"Safety r {sz_rad:.0f} m  (4x flame)"
        )

        # FIRELINE RECOMMENDATIONS panel — compact 3-line-per-candidate format
        fl_txt = [f"Attack: {atk_str}\n"]
        for c in candidates:
            # Wrap the reason text to the panel width (~40 chars) ourselves
            # rather than relying on matplotlib auto-wrap, which can clip.
            reason = c["reason"]
            wrap_at = 40
            words = reason.split()
            wrapped_lines, cur = [], ""
            for w in words:
                if len(cur) + len(w) + 1 > wrap_at:
                    wrapped_lines.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                wrapped_lines.append(cur)
            reason_block = "\n    ".join(wrapped_lines[:2])  # cap at 2 lines

            fl_txt.append(
                f"[{c['rank']}] {c['label']}  ({c['length']:.0f}m)\n"
                f"    {reason_block}"
            )
        self._txt_fireln.set_text("\n".join(fl_txt))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.04)

    # -- HELPERS --------------------------------------------------------------

    # -- PREDICT -------------------------------------------------------------
    def predict(self, dt: float, wind_speed: float, wind_dir: float, slope: float):
        self.wind_speed = wind_speed
        self.wind_dir   = wind_dir
        dx, dy, da, db, theta = self.spread.compute_increments(
            wind_speed, wind_dir, slope, dt)
        self.kf.predict(dx, dy, da, db, theta)
        self.current_time += dt
        self._last_ros = da / dt if dt > 0 else 0.0
        self._record()

    # -- UPDATE --------------------------------------------------------------
    def update(self, obs: DroneObservation):
        obs.timestamp = self.current_time
        self.observations.append(obs)
        self.kf.update(obs)

    def _record(self):
        kf  = self.kf
        ros = getattr(self, "_last_ros", 0.0)
        self.intensity_history.append(self.fireline.byram_intensity(ros))
        self.history.append({
            "t": self.current_time,
            "a": kf.semi_major,
            "b": kf.semi_minor,
            "area": kf.area_m2,
        })
        risk = self.risk.assess(kf, self.terrain, self.wind_speed, self.current_time)
        self.log_records.append([
            self.current_time, kf.cx, kf.cy, kf.semi_major, kf.semi_minor,
            f"{kf.area_m2/10_000:.4f}", f"{kf.confidence_pct:.1f}", risk,
        ])

    def export_log(self, path: Optional[str] = None):
        if not path:
            path = f"ferda_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "cx_m", "cy_m", "semi_major_m", "semi_minor_m",
                         "area_ha", "confidence_pct", "risk_level"])
            w.writerows(self.log_records)
        print(f"  OK Log exported → {path}")


# -----------------------------------------------------------------------------
# CLI HELPERS
# -----------------------------------------------------------------------------
def get_float(prompt: str, default: Optional[float] = None) -> float:
    while True:
        try:
            raw = input(prompt).strip()
            if not raw and default is not None:
                return default
            return float(raw)
        except ValueError:
            print("  [!] Numeric input required.")


def print_banner():
    print("\n" + "═" * 66)
    print("  FERDA v2.0  —  Fire Emergency Response Drone Analysis")
    print("  IndianaMap Agriculture Data Integration")
    print("  APIs: OSM Overpass, USDA SDM, USGS NHD, OpenTopoData, Open-Meteo")
    print("═" * 66)

def print_controls():
    print("\n  +- Runtime Commands -------------------------------------------+")
    print("  |  <Enter>              → auto-step (dt=10 s)                   |")
    print("  |  x,y,r               → fuse drone observation (Drone 0)       |")
    print("  |  x,y,r,id            → multi-drone (Drone id)                 |")
    print("  |  x,y,r,id,conf       → with confidence 0–1                    |")
    print("  |  env                  → reconfigure wind / slope              |")
    print("  |  export               → write CSV log                         |")
    print("  |  quit / Ctrl-C        → exit                                  |")
    print("  +--------------------------------------------------------------+\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print_banner()

    # -- Step 1: Location -----------------------------------------------------
    print("\n  [1/3]  Incident location (Indiana WGS84 coordinates)")
    print("         Default: West Lafayette / Tippecanoe County")
    lat = get_float("  Latitude  [40.4259] : ", default=40.4259)
    lon = get_float("  Longitude [-86.9081]: ", default=-86.9081)

    # -- Step 2: IndianaMap terrain fetch -------------------------------------
    print("\n  [2/3]  Fetching terrain data from IndianaMap...")
    client  = IndianaMapClient(verbose=True)
    terrain = client.build_terrain_profile(lat, lon)

    print("  Override effective terrain multiplier? (blank = keep IndianaMap value)")
    ov = input(f"  Terrain Multiplier [{terrain.effective_mult():.3f}]: ").strip()
    if ov:
        try:
            terrain.fuel_mult    = float(ov)
            terrain.soil_moisture = 0.0   # reset moisture correction so mult is literal
        except ValueError:
            pass

    # -- Step 3: Fire initialisation ------------------------------------------
    print("\n  [3/3]  Initial fire parameters")
    ix = get_float("  Initial X (m) [0]   : ", default=0.0)
    iy = get_float("  Initial Y (m) [0]   : ", default=0.0)
    ir = get_float("  Initial Radius (m) [5]: ", default=5.0)

    ferda = FERDA_System(ix, iy, ir, terrain)
    print_controls()

    try:
        while True:
            # -- Environment config --------------------------------------------
            print("\n  --- Environment (weather auto-fetched from Open-Meteo) ------")
            _client = IndianaMapClient(verbose=False)
            wx = _client.query_weather(ferda.terrain.lat, ferda.terrain.lon)
            ferda.terrain.weather = wx
            print(f"  Weather  : {wx.description}")
            print(f"  Wind     : {wx.wind_speed:.1f} m/s from {wx.wind_dir_met:.0f}deg-met"
                  f" -> {wx.wind_dir_math:.0f}deg-math")
            print(f"  Humidity : {wx.humidity:.0f}%  |  Temp: {wx.temperature:.1f}C"
                  f"  |  Precip: {wx.precipitation:.1f}mm/h")
            ws_raw = input(f"  Wind speed  [Enter=keep {wx.wind_speed:.1f} m/s]: ").strip()
            wd_raw = input(f"  Wind dir    [Enter=keep {wx.wind_dir_math:.0f} deg-math]: ").strip()
            wind_v = float(ws_raw) if ws_raw else wx.wind_speed
            wind_d = float(wd_raw) if wd_raw else wx.wind_dir_math
            auto_slope = ferda.terrain.slope_deg
            sl_raw = input(f"  Slope       [Enter=keep {auto_slope:.1f} deg]: ").strip()
            slope = float(sl_raw) if sl_raw else auto_slope

            # -- Simulation loop -----------------------------------------------
            print("\n  --- Simulation running (type 'env' to reconfigure) ----------")
            while True:
                ferda.predict(dt=10.0, wind_speed=wind_v,
                              wind_dir=wind_d, slope=slope)
                ferda.render()

                cmd = input(
                    f"  t={ferda.current_time:.0f}s  → "
                ).strip().lower()

                if not cmd:
                    continue

                if cmd in ("q", "quit", "exit"):
                    raise KeyboardInterrupt

                if cmd == "env":
                    break

                if cmd == "export":
                    ferda.export_log()
                    continue

                # Parse observation
                try:
                    parts = [float(v) for v in cmd.split(",")]
                    if len(parts) < 3:
                        print("  [!] Format: x,y,r  or  x,y,r,drone_id  or  x,y,r,id,conf")
                        continue
                    obs = DroneObservation(
                        x=parts[0], y=parts[1], radius=parts[2],
                        drone_id=int(parts[3]) if len(parts) > 3 else 0,
                        confidence=float(parts[4]) if len(parts) > 4 else 1.0,
                    )
                    ferda.update(obs)
                    ferda.render()
                    print(f"  OK Observation from Drone {obs.drone_id} fused "
                          f"(conf={obs.confidence:.1f})")
                except (ValueError, IndexError):
                    print("  [!] Numeric input required.")

    except KeyboardInterrupt:
        print("\n\n  Simulation ended — exporting log...")
        ferda.export_log()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
