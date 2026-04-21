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
    "farmland": 82, "farmyard": 82, "orchard": 82, "vineyard": 82,
    "meadow":   81, "grass": 71, "recreation_ground": 71,
    "village_green": 71, "cemetery": 71,
    "forest":   41, "logging": 41,
    "residential": 22, "commercial": 23, "industrial": 23, "retail": 22,
    "construction": 31, "brownfield": 31, "quarry": 31, "landfill": 31,
    "wood":     41, "scrub": 52, "heath": 52, "grassland": 71,
    "wetland":  90, "marsh": 95, "water": 11,
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
    risk_level:        str   = "medium"
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
class AlertEvent:
    time:    float
    level:   str    # "INFO" | "WARNING" | "CRITICAL"
    message: str


# -----------------------------------------------------------------------------
# INDIANAMAP CLIENT
# -----------------------------------------------------------------------------
class IndianaMapClient:
    """
    Fetches terrain/agriculture data from publicly accessible federal APIs:

      1. USDA CropScape (NASS)  — crop/land-cover type at a lat/lon point
         https://www.mrlc.gov/geoserver/NLCD_Land_Cover/wms
         CDL codes differ from NLCD; see CDL_FUEL_MODELS below.

      2. USDA SSURGO via SDM Tabular Service (NRCS)  — soil drainage class
         https://SDMDataAccess.nrcs.usda.gov/tabular/post.rest
         Spatial query: map-unit polygon that contains the point.

      3. USGS StreamStats / NHD  — water body proximity
         https://streamstats.usgs.gov/regressionservices/

      4. USGS EPQS (updated URL)  — elevation
         https://epqs.nationalmap.gov/v1/json

    All endpoints are free, public-domain, no API key required.
    maps.indiana.edu has been removed — that server is offline.
    """

    # Source A: USGS/MRLC NLCD 2021 WMS GetFeatureInfo
    # Source: https://www.mrlc.gov/data-services-page
    MRLC_WMS_URL  = "https://www.mrlc.gov/geoserver/NLCD_Land_Cover/wms"
    MRLC_LAYER    = "NLCD_2021_Land_Cover_L48"

    # Source B: ESRI Living Atlas NLCD 2021 ImageServer — different CDN/host
    # Public ArcGIS Image Service, no API key needed.
    # If MRLC GeoServer is blocked, this hits Esri's CDN instead.
    ESRI_NLCD_URL = "https://landscape1.arcgis.com/arcgis/rest/services/USA_NLCD_2021/ImageServer/identify"
    OVERPASS_URL  = "https://overpass-api.de/api/interpreter"

    # USDA SSURGO / Soil Data Mart tabular REST  (accepts SQL, returns JSON)
    SDM_URL       = "https://SDMDataAccess.nrcs.usda.gov/tabular/post.rest"

    # USGS Elevation Point Query Service (updated URL, replaces nationalmap.gov/epqs)
    USGS_ELEV     = "https://epqs.nationalmap.gov/v1/json"

    # USGS NHD REST  (National Hydrography Dataset — water bodies)
    NHD_URL       = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"

    TIMEOUT = 6    # seconds — fail fast; all three sources have reliable fallbacks

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cache: dict = {}
        self._cdl_failed  = False   # set True when MRLC WMS is unreachable
        self._soil_failed  = False   # set True when SDM is unreachable

    def _log(self, msg: str):
        if self.verbose:
            print(f"  |  {msg}")

    # -- 1. Land Cover — waterfall across two independent endpoints -----------
    def query_land_cover(self, lat: float, lon: float) -> int:
        """
        Tries two completely independent NLCD sources in sequence:

          Source A: USGS/MRLC GeoServer WMS GetFeatureInfo
                    mrlc.gov/geoserver — USGS operated
          Source B: ESRI Living Atlas NLCD 2021 ImageServer
                    landscape1.arcgis.com — Esri CDN, different network path

        Both return NLCD 2021 land cover codes. Using two sources means a
        single network block or outage won't reach the manual picker.
        """
        cache_key = f"nlcd_{lat:.4f}_{lon:.4f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # ── Source A: MRLC WMS ────────────────────────────────────────────
        self._log("Querying land cover [1/3] USGS MRLC WMS...")
        delta = 0.005
        bbox  = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
        try:
            r = requests.get(
                self.MRLC_WMS_URL,
                params={
                    "SERVICE":      "WMS",
                    "VERSION":      "1.1.1",
                    "REQUEST":      "GetFeatureInfo",
                    "LAYERS":       self.MRLC_LAYER,
                    "QUERY_LAYERS": self.MRLC_LAYER,
                    "STYLES":       "",
                    "BBOX":         bbox,
                    "WIDTH":        "11",
                    "HEIGHT":       "11",
                    "SRS":          "EPSG:4326",
                    "X":            "5",
                    "Y":            "5",
                    "INFO_FORMAT":  "application/json",
                    "FEATURE_COUNT":"1",
                },
                timeout=self.TIMEOUT,
            )
            features = r.json().get("features", [])
            if features:
                props = features[0].get("properties", {})
                for k in ("GRAY_INDEX", "value", "pixel"):
                    if k in props:
                        code = int(props[k])
                        if code in NLCD_FUEL_MODELS:
                            self._log(f"NLCD {code}: {NLCD_FUEL_MODELS[code]['name']} (MRLC)")
                            self._cache[cache_key] = code
                            return code
        except Exception as e:
            self._log(f"MRLC WMS failed ({type(e).__name__}) — trying fallback...")

        # ── Source B: ESRI Living Atlas ImageServer identify ─────────────
        self._log("Querying land cover [2/3] ESRI Living Atlas NLCD...")
        try:
            r = requests.get(
                self.ESRI_NLCD_URL,
                params={
                    "geometry":      f"{lon},{lat}",
                    "geometryType":  "esriGeometryPoint",
                    "sr":            "4326",
                    "returnGeometry":"false",
                    "f":             "json",
                },
                timeout=self.TIMEOUT,
            )
            data  = r.json()
            # ImageServer identify returns {"value": "82", "name": "Cultivated Crops"}
            value = data.get("value") or data.get("catalogItems", {})
            if value and str(value).strip() not in ("", "NoData", "Null"):
                code = int(float(str(value).strip()))
                if code in NLCD_FUEL_MODELS:
                    self._log(f"NLCD {code}: {NLCD_FUEL_MODELS[code]['name']} (ESRI Atlas)")
                    self._cache[cache_key] = code
                    return code
        except Exception as e:
            self._log(f"ESRI Atlas failed ({type(e).__name__})")

        # -- Source C: OpenStreetMap Overpass API --------------------------
        # Tries two Overpass mirrors in sequence with a User-Agent header.
        # Overpass silently rate-limits anonymous requests; a descriptive
        # User-Agent is polite and avoids most blocks.
        OVERPASS_MIRRORS = [
            self.OVERPASS_URL,                          # overpass-api.de (main)
            "https://overpass.kumi.systems/api/interpreter",  # kumi.systems mirror
        ]
        HEADERS = {"User-Agent": "FERDA-FireDetection/2.0 (drone fire spread tracker)"}
        query = (
            f"[out:json][timeout:8];"
            f"is_in({lat},{lon})->.a;"
            f"(area.a[landuse];area.a[natural];);"
            f"out tags;"
        )
        for mirror_url in OVERPASS_MIRRORS:
            self._log(f"Querying land cover [3/3] OpenStreetMap Overpass ({mirror_url.split('/')[2]})...")
            try:
                r = requests.post(
                    mirror_url,
                    data={"data": query},
                    headers=HEADERS,
                    timeout=self.TIMEOUT,
                )
                # Guard: some error responses return HTML instead of JSON
                ct = r.headers.get("Content-Type", "")
                if r.status_code != 200 or "html" in ct.lower():
                    self._log(f"Overpass non-JSON response (HTTP {r.status_code}) — trying next mirror")
                    continue
                elements = r.json().get("elements", [])
                for el in reversed(elements):  # reversed = smallest/most specific area first
                    tags = el.get("tags", {})
                    for osm_key in ("landuse", "natural"):
                        val = tags.get(osm_key, "").lower()
                        if val in OSM_TO_NLCD:
                            code = OSM_TO_NLCD[val]
                            self._log(f"OSM {osm_key}={val} -> NLCD {code}: "
                                      f"{NLCD_FUEL_MODELS[code]['name']} (Overpass)")
                            self._cache[cache_key] = code
                            return code
                self._log("Overpass: no landuse/natural tag found at this point")
                break   # got a valid response, just no matching tag — no point trying mirror
            except Exception as e:
                self._log(f"Overpass mirror failed ({type(e).__name__}) — trying next")

        # -- All three failed -> manual picker -----------------------------
        self._cdl_failed = True
        self._log("All 3 land cover sources unreachable -- will prompt manually")
        return DEFAULT_NLCD

    # -- 2. Soil data via USDA SDM Tabular REST --------------------------------
    def query_soils(self, lat: float, lon: float) -> dict:
        """
        Query USDA Soil Data Mart (SDM) for the dominant soil component at point.
        Uses a spatial SQL query against the SSURGO mapunit table.
        Returns drainage class, hydric flag, and a moisture proxy.
        """
        self._log("Querying soils (USDA SSURGO / SDM Tabular)...")
        defaults = {
            "name": "Silty Clay Loam (Indiana default)",
            "drainage": "Well drained",
            "hydric": "N",
            "moisture": 0.35,
        }

        # SDM tabular SQL: find dominant component for the map unit containing point
        sql = (
            f"SELECT TOP 1 co.compname, co.drainagecl, co.hydricrating "
            f"FROM mapunit mu "
            f"INNER JOIN component co ON mu.mukey = co.mukey "
            f"INNER JOIN mupolygon mp ON mu.mukey = mp.mukey "
            f"WHERE co.majcompflag = 'Yes' "
            f"AND mp.mupolygongeo.STContains("
            f"geometry::STGeomFromText('POINT({lon} {lat})', 4326)) = 1 "
            f"ORDER BY co.comppct_r DESC"
        )
        try:
            r = requests.post(
                self.SDM_URL,
                data={"format": "JSON+COLUMNNAME", "query": sql},
                timeout=self.TIMEOUT,
            )
            rows = r.json().get("Table", [])
            if len(rows) >= 2:   # row 0 = column names, row 1 = first data row
                cols = rows[0]
                vals = rows[1]
                row  = dict(zip(cols, vals))
                name   = str(row.get("compname",    defaults["name"]))
                drain  = str(row.get("drainagecl",  defaults["drainage"]))
                hydric = str(row.get("hydricrating", "No"))
                hydric_flag = "Y" if hydric.lower() in ("yes", "y") else "N"

                drain_moisture = {
                    "Excessively drained":          0.05,
                    "Somewhat excessively drained": 0.12,
                    "Well drained":                 0.25,
                    "Moderately well drained":      0.40,
                    "Somewhat poorly drained":      0.55,
                    "Poorly drained":               0.70,
                    "Very poorly drained":          0.85,
                }
                moisture = drain_moisture.get(drain, 0.35)
                self._log(f"Soil: {name} | {drain} | Hydric: {hydric_flag} | Moisture: {moisture:.0%}")
                return {"name": name, "drainage": drain, "hydric": hydric_flag, "moisture": moisture}
        except Exception as e:
            self._log(f"SDM unreachable: {type(e).__name__}")
            self._soil_failed = True

        self._log("Using Indiana default soil profile")
        return defaults

    # -- 3. Water body proximity via USGS NHD ----------------------------------
    def query_water_nearby(self, lat: float, lon: float, buffer_deg: float = 0.005) -> bool:
        """
        Query USGS National Hydrography Dataset REST service for water bodies
        within ~500 m of the point (0.005° ≈ 500 m at Indiana latitudes).
        """
        self._log("Checking water bodies (USGS NHD)...")
        xmin = lon - buffer_deg; xmax = lon + buffer_deg
        ymin = lat - buffer_deg; ymax = lat + buffer_deg
        try:
            r = requests.get(
                f"{self.NHD_URL}/0/query",    # layer 0 = NHDArea (water bodies)
                params={
                    "geometry":     f"{xmin},{ymin},{xmax},{ymax}",
                    "geometryType": "esriGeometryEnvelope",
                    "inSR":         "4326",
                    "spatialRel":   "esriSpatialRelIntersects",
                    "returnCountOnly": "true",
                    "f":            "json",
                },
                timeout=self.TIMEOUT,
            )
            count = r.json().get("count", 0)
            found = count > 0
            self._log(f"Water bodies found: {'YES ({count} features — spread reduced)' if found else 'No'}")
            return found
        except Exception as e:
            self._log(f"NHD water query failed: {e}")
            return False

    # -- 4. Elevation via USGS EPQS --------------------------------------------
    def query_elevation(self, lat: float, lon: float) -> Optional[float]:
        """USGS Elevation Point Query Service — returns elevation in metres."""
        self._log("Querying elevation (USGS EPQS)...")
        try:
            r = requests.get(
                self.USGS_ELEV,
                params={"x": lon, "y": lat, "units": "Meters",
                        "wkid": "4326", "includeDate": "false"},
                timeout=self.TIMEOUT,
            )
            elev = float(r.json()["value"])
            self._log(f"Elevation: {elev:.1f} m ASL")
            return elev
        except Exception as e:
            self._log(f"Elevation query failed: {e}")
            return None

    # -- Assemble TerrainProfile -----------------------------------------------
    # ── Interactive terrain picker (used when APIs are unreachable) ──────────
    @staticmethod
    def _pick_land_cover() -> int:
        """Show a numbered menu of common terrain types and return NLCD code."""
        MENU = [
            (82,  "Cultivated Crops       (mult x1.10 — medium)"),
            (81,  "Hay / Pasture          (mult x1.30 — medium-high)"),
            (71,  "Grassland / Herbaceous (mult x1.40 — high)"),
            (52,  "Shrub / Scrub          (mult x1.60 — high)"),
            (41,  "Deciduous Forest       (mult x1.20 — medium)"),
            (42,  "Evergreen Forest       (mult x1.80 — CRITICAL)"),
            (43,  "Mixed Forest           (mult x1.50 — high)"),
            (31,  "Barren / Fallow        (mult x0.10 — low)"),
            (11,  "Open Water / Wetland   (mult x0.00 — no spread)"),
            (21,  "Developed / Open Space (mult x0.30 — low)"),
        ]
        print("\n  +-- Land Cover / Fuel Model (API unavailable) --------------------")
        for i, (code, label) in enumerate(MENU, 1):
            print(f"  |  {i:2d}. {label}")
        print("  +------------------------------------------------------------------")
        while True:
            raw = input("  Select [1]: ").strip()
            if not raw:
                return MENU[0][0]  # Cultivated Crops default
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(MENU):
                    return MENU[idx][0]
            except ValueError:
                pass
            print("  [!] Enter a number 1–10")

    @staticmethod
    def _pick_soil() -> dict:
        """Show a drainage-class menu and return a soil dict."""
        MENU = [
            ("Well drained",           0.25, "N", "Sandy Loam / Silt Loam"),
            ("Moderately well drained", 0.40, "N", "Silt Loam / Loam"),
            ("Somewhat poorly drained", 0.55, "N", "Silty Clay Loam"),
            ("Poorly drained",          0.70, "Y", "Silty Clay / Muck (hydric)"),
            ("Excessively drained",     0.05, "N", "Loamy Sand / Gravelly"),
        ]
        print("\n  +-- Soil Drainage Class (API unavailable) ------------------------")
        for i, (drain, moist, hydric, name) in enumerate(MENU, 1):
            hydric_note = "  [hydric]" if hydric == "Y" else ""
            print(f"  |  {i}. {drain:<32} moisture ~{moist:.0%}{hydric_note}")
        print("  +------------------------------------------------------------------")
        while True:
            raw = input("  Select [1]: ").strip()
            if not raw:
                choice = MENU[0]
                break
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(MENU):
                    choice = MENU[idx]
                    break
            except ValueError:
                pass
            print("  [!] Enter a number 1–5")
        drain, moist, hydric, name = choice
        return {"name": name, "drainage": drain, "hydric": hydric, "moisture": moist}

    def build_terrain_profile(self, lat: float, lon: float) -> TerrainProfile:
        """
        Queries USGS MRLC NLCD WMS, USDA SDM, and USGS NHD.
        If any service is unreachable, drops into an interactive picker
        so the operator always gets a meaningful terrain profile.
        """
        print(f"\n  +--- Terrain Fetch (USDA/USGS APIs) --- ({lat:.4f}N, {abs(lon):.4f}W) --")

        nlcd = self.query_land_cover(lat, lon)
        soil = self.query_soils(lat, lon)
        water = self.query_water_nearby(lat, lon)

        # ── Manual override when APIs were unreachable ─────────────────────
        if self._cdl_failed:
            print("  [!] Land cover API offline — please select terrain manually:")
            nlcd = self._pick_land_cover()

        if self._soil_failed:
            print("  [!] Soil API offline — please select drainage class manually:")
            soil = self._pick_soil()

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
            risk_level      = fuel_data["risk"],
            lat=lat, lon=lon,
        )
        print(f"  |")
        print(f"  |  * Land Cover  : {profile.land_cover_name}  (NLCD {nlcd})")
        print(f"  |  * Base Mult    : {profile.fuel_mult:.2f}  ->  Effective: {profile.effective_mult():.2f}")
        print(f"  |  * Soil         : {profile.soil_name[:40]}")
        print(f"  |  * Drainage     : {profile.drainage_class}  (moisture {profile.soil_moisture:.0%})")
        print(f"  |  * Hydric Soil  : {profile.hydric_rating}  (wetland penalty if Y)")
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

        ros = self.BASE_ROS * (1 + phi_w + phi_s) * eff   # m/s

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
        self.txt_status.set_text(
            f"[t]  {self.current_time:.0f} s\n"
            f"Conf  {conf:.1f}%   Risk: {risk_lvl.upper()}\n"
            f"NIS   {nis_str}  (expect < 7.8)\n"
            f"CX {kf.cx:.1f} m   CY {kf.cy:.1f} m\n"
            f"a={kf.semi_major:.1f}m  b={kf.semi_minor:.1f}m\n"
            f"Area  {kf.area_m2/10_000:.3f} ha\n"
            f"Wind  {self.wind_speed:.1f} m/s @ {self.wind_dir:.0f}°"
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
    print("  Source: https://www.indianamap.org  (maps.indiana.edu REST API)")
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
            print("\n  --- Environment ---------------------------------------------")
            wind_v = get_float("  Wind Speed (m/s): ")
            wind_d = get_float("  Wind Direction (° from East, CCW): ")
            slope  = get_float("  Slope Angle (°): ")

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
