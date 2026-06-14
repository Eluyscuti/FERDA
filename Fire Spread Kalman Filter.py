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
# FERDA v2.0 — MAIN SYSTEM
# -----------------------------------------------------------------------------
class FERDA_System:
    def __init__(self, ix: float, iy: float, ir: float, terrain: TerrainProfile):
        self.kf           = FireKalmanFilter(ix, iy, ir)
        self.spread       = FireSpreadModel(terrain)
        self.terrain      = terrain
        self.risk         = RiskAssessor()
        self.current_time = 0.0
        self.wind_speed   = 0.0
        self.wind_dir     = 0.0

        self.history: List[dict] = [{"t": 0, "a": ir, "b": ir * 0.65,
                                      "area": math.pi * ir * ir * 0.65}]
        self.observations: List[DroneObservation] = []
        self.log_records:  List[list] = []

        self._build_figure()

    # -- FIGURE ---------------------------------------------------------------
    def _build_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(19, 9), facecolor="#13132a")
        gs = self.fig.add_gridspec(
            2, 3, left=0.05, right=0.97, top=0.93, bottom=0.07,
            hspace=0.40, wspace=0.30,
            width_ratios=[2.2, 1, 1],
        )
        self.ax_map    = self.fig.add_subplot(gs[:, 0])   # Main fire map
        self.ax_growth = self.fig.add_subplot(gs[0, 1])   # Axis growth chart
        self.ax_area   = self.fig.add_subplot(gs[1, 1])   # Area / ha chart
        self.ax_alerts = self.fig.add_subplot(gs[:, 2])   # Alerts / info panel

        dark_bg = "#0d0d1f"
        for ax in (self.ax_map, self.ax_growth, self.ax_area, self.ax_alerts):
            ax.set_facecolor(dark_bg)
            for sp in ax.spines.values():
                sp.set_color("#33335a")
            ax.tick_params(colors="#9999bb", labelsize=7)
            ax.xaxis.label.set_color("#9999bb")
            ax.yaxis.label.set_color("#9999bb")

        # -- Map panel --------------------------------------------------------
        SPAN = 650
        self.ax_map.set_xlim(-80, SPAN)
        self.ax_map.set_ylim(-80, SPAN)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_xlabel("X (m)", fontsize=8)
        self.ax_map.set_ylabel("Y (m)", fontsize=8)
        self.ax_map.grid(True, alpha=0.12, color="#33335a", linewidth=0.6)

        # Terrain info banner
        risk_col = RISK_COLORS.get(self.terrain.risk_level, "#FFC000")
        self.ax_map.text(
            0.01, 0.995,
            f"[crop] {self.terrain.land_cover_name}  ·  Soil: {self.terrain.soil_name[:28]}\n"
            f"Fuel ×{self.terrain.fuel_mult:.1f}  ·  Effective ×{self.terrain.effective_mult():.2f}"
            f"  ·  Moisture {self.terrain.soil_moisture:.0%}"
            + ("  ·  ! Water barrier" if self.terrain.has_water_nearby else ""),
            transform=self.ax_map.transAxes, fontsize=7.8, va="top",
            color="#ccccee",
            bbox=dict(facecolor="#1e1e3a", edgecolor=risk_col, alpha=0.92, pad=3),
        )

        # Fire ellipse
        ix, iy = self.kf.cx, self.kf.cy
        self.patch_fire = Ellipse(
            (ix, iy), self.kf.semi_major * 2, self.kf.semi_minor * 2,
            angle=0, color="#FF4500", alpha=0.72, zorder=5,
        )
        self.ax_map.add_patch(self.patch_fire)

        # Uncertainty ellipse (2-sigma shell)
        self.patch_uncert = Ellipse(
            (ix, iy), self.kf.semi_major * 2 + 12, self.kf.semi_minor * 2 + 12,
            angle=0, edgecolor="#FFD700", facecolor="none",
            alpha=0.45, linestyle="--", linewidth=1.2, zorder=4,
        )
        self.ax_map.add_patch(self.patch_uncert)

        # Wind quiver (top-right corner)
        self.quiver = self.ax_map.quiver(
            SPAN - 60, SPAN - 60, 0, 0,
            color="#00BFFF", scale=250, width=0.006,
            headwidth=4, headlength=5, zorder=7,
        )
        self.ax_map.text(SPAN - 60, SPAN - 30, "Wind", color="#00BFFF",
                         fontsize=7, ha="center", va="bottom")

        # Status text box
        self.txt_status = self.ax_map.text(
            0.01, 0.86, "", transform=self.ax_map.transAxes,
            fontsize=8.5, fontweight="bold", va="top", color="white",
            bbox=dict(facecolor="#1e1e3a", edgecolor="#444466", alpha=0.90, pad=4),
        )

        # Per-drone observation markers (populated on first observation)
        self.obs_artists: dict = {}

        # Legend
        legend_patches = [
            mpatches.Patch(color="#FF4500", alpha=0.7,  label="Fire Perimeter"),
            mpatches.Patch(edgecolor="#FFD700", facecolor="none",
                           linestyle="--", linewidth=1.2, label="2σ Uncertainty"),
        ]
        self.ax_map.legend(
            handles=legend_patches, loc="lower right", fontsize=7,
            facecolor="#1e1e3a", edgecolor="#444466", labelcolor="#ccccee",
        )

        # -- Growth chart -----------------------------------------------------
        self.ax_growth.set_title("Fire Axes (m)", color="#ccccee", fontsize=8, pad=3)
        self.ax_growth.set_xlabel("Time (s)", fontsize=7)
        self.ax_growth.grid(True, alpha=0.18, color="#33335a")
        self.ln_major, = self.ax_growth.plot([], [], color="#FF4500", lw=1.6, label="Semi-major")
        self.ln_minor, = self.ax_growth.plot([], [], color="#FFD700", lw=1.2,
                                              linestyle="--", label="Semi-minor")
        self.ax_growth.legend(facecolor="#0d0d1f", edgecolor="#33335a",
                               labelcolor="#ccccee", fontsize=6.5)

        # -- Area chart -------------------------------------------------------
        self.ax_area.set_title("Fire Area (ha)", color="#ccccee", fontsize=8, pad=3)
        self.ax_area.set_xlabel("Time (s)", fontsize=7)
        self.ax_area.grid(True, alpha=0.18, color="#33335a")
        self.ln_area, = self.ax_area.plot([], [], color="#9B00FF", lw=1.5)

        # -- Alerts panel ------------------------------------------------------
        self.ax_alerts.axis("off")
        self.ax_alerts.set_title("System Alerts & State", color="#ccccee",
                                  fontsize=8.5, pad=6)
        self.txt_alerts = self.ax_alerts.text(
            0.04, 0.97, "", transform=self.ax_alerts.transAxes,
            fontsize=6.8, va="top", color="#ccccee", fontfamily="monospace",
            linespacing=1.55,
        )

    # -- PREDICT --------------------------------------------------------------
    def predict(self, dt: float, wind_speed: float, wind_dir: float, slope: float):
        self.wind_speed = wind_speed
        self.wind_dir   = wind_dir
        dx, dy, da, db, theta = self.spread.compute_increments(
            wind_speed, wind_dir, slope, dt)
        self.kf.predict(dx, dy, da, db, theta)
        self.current_time += dt
        self._record()

    # -- UPDATE ---------------------------------------------------------------
    def update(self, obs: DroneObservation):
        obs.timestamp = self.current_time
        self.observations.append(obs)
        self.kf.update(obs)

    # -- RENDER ---------------------------------------------------------------
    def render(self):
        kf  = self.kf
        ts  = [h["t"] for h in self.history]
        mas = [h["a"] for h in self.history]
        mis = [h["b"] for h in self.history]
        ars = [h["area"] / 10_000 for h in self.history]

        # Risk assessment
        risk_lvl = self.risk.assess(kf, self.terrain, self.wind_speed, self.current_time)
        fire_col = RISK_COLORS.get(risk_lvl, "#FF4500")

        # Fire ellipse
        self.patch_fire.center  = (kf.cx, kf.cy)
        self.patch_fire.width   = kf.semi_major * 2
        self.patch_fire.height  = kf.semi_minor * 2
        self.patch_fire.angle   = math.degrees(kf.orientation_rad)
        self.patch_fire.set_facecolor(fire_col)

        # Uncertainty ellipse — scale with positional uncertainty
        pos_std = math.sqrt(kf.P[0, 0] + kf.P[1, 1])
        self.patch_uncert.center = (kf.cx, kf.cy)
        self.patch_uncert.width  = kf.semi_major * 2 + pos_std * 2
        self.patch_uncert.height = kf.semi_minor * 2 + pos_std * 2
        self.patch_uncert.angle  = math.degrees(kf.orientation_rad)

        # Ghost trail
        ghost = Ellipse(
            (kf.cx, kf.cy), kf.semi_major * 2, kf.semi_minor * 2,
            angle=math.degrees(kf.orientation_rad),
            edgecolor=fire_col, facecolor="none", alpha=0.07,
            linestyle=":", linewidth=0.8, zorder=3,
        )
        self.ax_map.add_patch(ghost)

        # Wind arrow
        theta = math.radians(self.wind_dir)
        self.quiver.set_UVC(
            math.cos(theta) * self.wind_speed,
            math.sin(theta) * self.wind_speed,
        )

        # Drone observation markers
        drone_colors = plt.cm.Set2.colors
        for obs in self.observations[-20:]:
            did = obs.drone_id
            col = drone_colors[did % len(drone_colors)]
            if did not in self.obs_artists:
                ln, = self.ax_map.plot(
                    [], [], marker="x", markersize=9, linewidth=0,
                    color=col, zorder=8, markeredgewidth=2,
                    label=f"Drone {did}",
                )
                self.obs_artists[did] = ln
            pts = [(o.x, o.y) for o in self.observations if o.drone_id == did]
            xs, ys = zip(*pts)
            self.obs_artists[did].set_data(xs, ys)

        # Growth charts
        self.ln_major.set_data(ts, mas)
        self.ln_minor.set_data(ts, mis)
        self.ln_area.set_data(ts, ars)
        for ax in (self.ax_growth, self.ax_area):
            ax.relim(); ax.autoscale_view()

        # Status text
        conf    = kf.confidence_pct
        c_color = "#00FF7F" if conf > 70 else "#FFA500" if conf > 40 else "#FF4444"
        nis_str = f"{kf.last_nis:.1f}" if kf.last_nis is not None else "n/a"
        wx = getattr(self.terrain, "weather", None)
        wx_line = (f"{wx.description} | RH:{wx.humidity:.0f}% | T:{wx.temperature:.1f}C"
                   if wx and wx.wind_speed > 0 else "Weather: not fetched")
        self.txt_status.set_text(
            f"[t]  {self.current_time:.0f} s\n"
            f"Conf  {conf:.1f}%   Risk: {risk_lvl.upper()}\n"
            f"NIS   {nis_str}  (expect < 7.8)\n"
            f"CX {kf.cx:.1f} m   CY {kf.cy:.1f} m\n"
            f"a={kf.semi_major:.1f}m  b={kf.semi_minor:.1f}m\n"
            f"Area  {kf.area_m2/10_000:.3f} ha\n"
            f"Wind  {self.wind_speed:.1f} m/s @ {self.wind_dir:.0f}\u00b0\n"
            f"{wx_line}"
        )
        self.txt_status.set_color(c_color)

        # Alerts panel
        recent = self.risk.alerts[-10:]
        lines = [f"[{a.level:8s}] t={a.time:5.0f}s\n  {a.message[:42]}"
                 for a in reversed(recent)]
        self.txt_alerts.set_text("\n".join(lines) if lines else "No alerts yet.")

        # Title
        self.ax_map.set_title(
            f"FERDA v2.0  ·  {self.terrain.land_cover_name}"
            f"  ·  ({self.terrain.lat:.4f}°N, {abs(self.terrain.lon):.4f}°W)",
            color="#ddddff", fontsize=10, fontweight="bold",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.04)

    # -- HELPERS --------------------------------------------------------------
    def _record(self):
        kf = self.kf
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
