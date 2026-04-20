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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import requests
import json
import csv
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# NLCD FUEL MODEL TABLE
# Maps National Land Cover Database codes to Rothermel-style fire parameters.
# Critical for Indiana agricultural landscapes:
#   Code 81 = Hay/Pasture, 82 = Cultivated Crops (dominant in Indiana)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# INDIANAMAP CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class IndianaMapClient:
    """
    Queries live IndianaMap ArcGIS REST services for:
      • NLCD Land Cover 2006  → fuel model
      • SSURGO Soils 2019     → soil moisture / drainage
      • NHD Water Bodies      → firebreak detection

    All endpoints are public, no API key required.
    Coordinate system: EPSG:26916 (UTM Zone 16N) as required by maps.indiana.edu.
    """

    BASE = "https://maps.indiana.edu/arcgis/rest/services"

    # Confirmed REST endpoints from IndianaMap (maps.indiana.edu)
    LAND_COVER_SVC = f"{BASE}/Environment/Land_Cover_2006/MapServer"
    SOILS_SVC      = f"{BASE}/Environment/Soils_SSURGO_Soil_Survey/MapServer"
    WATER_SVC      = f"{BASE}/Hydrology/Water_Bodies_Lakes_LocalRes/MapServer"

    # USGS Elevation Point Query (national service, no auth needed)
    USGS_ELEV      = "https://nationalmap.gov/epqs/pqs.php"

    TIMEOUT = 12

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cache: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  │  {msg}")

    def _identify(self, service_url: str, utm_x: float, utm_y: float,
                  pad: float = 200.0) -> Optional[dict]:
        """Call ArcGIS Identify at UTM point, return first result or None."""
        params = {
            "geometry":      json.dumps({"x": utm_x, "y": utm_y,
                                         "spatialReference": {"wkid": 26916}}),
            "geometryType":  "esriGeometryPoint",
            "sr":            "26916",
            "layers":        "all",
            "tolerance":     "5",
            "mapExtent":     f"{utm_x-pad},{utm_y-pad},{utm_x+pad},{utm_y+pad}",
            "imageDisplay":  "400,400,96",
            "returnGeometry":"false",
            "f":             "json",
        }
        try:
            r = requests.get(f"{service_url}/identify", params=params, timeout=self.TIMEOUT)
            data = r.json()
            results = data.get("results", [])
            return results[0] if results else None
        except Exception as e:
            self._log(f"Identify failed ({service_url.split('/')[-2]}): {e}")
            return None

    def query_land_cover(self, lat: float, lon: float) -> int:
        """Return NLCD code at location from IndianaMap Land Cover service."""
        key = f"nlcd_{lat:.4f}_{lon:.4f}"
        if key in self._cache:
            return self._cache[key]

        self._log("Querying Land Cover (NLCD 2006)...")
        ux, uy = latlon_to_utm16n(lat, lon)
        result = self._identify(self.LAND_COVER_SVC, ux, uy)

        if result:
            attrs = result.get("attributes", {})
            # NLCD layer exposes pixel value under several possible field names
            for field in ("Pixel Value", "Value", "NLCD_Land", "gridcode", "CLASS"):
                if field in attrs:
                    try:
                        code = int(float(str(attrs[field]).strip()))
                        if code in NLCD_FUEL_MODELS:
                            self._log(f"Land cover → NLCD {code}: {NLCD_FUEL_MODELS[code]['name']}")
                            self._cache[key] = code
                            return code
                    except (ValueError, TypeError):
                        continue

        self._log(f"Land cover lookup miss — defaulting to NLCD {DEFAULT_NLCD} (Cultivated Crops)")
        return DEFAULT_NLCD

    def query_soils(self, lat: float, lon: float) -> dict:
        """
        Return soil attributes from IndianaMap SSURGO Soil Survey.
        Key attrs: MAPUNIT_NA (map-unit name), DRCLASSDCD (drainage class),
                   HYDCLPRS (hydric rating), FORPEHRTDC (erosion hazard).
        """
        self._log("Querying Soils (SSURGO 2019)...")
        ux, uy = latlon_to_utm16n(lat, lon)
        result = self._identify(self.SOILS_SVC, ux, uy)

        defaults = {
            "name": "Silty Clay Loam (Indiana default)",
            "drainage": "Well drained",
            "hydric": "N",
            "moisture": 0.35,
        }

        if not result:
            return defaults

        attrs = result.get("attributes", {})
        name    = str(attrs.get("MAPUNIT_NA", attrs.get("Soil Name", "Unknown"))).strip()
        drain   = str(attrs.get("DRCLASSDCD", "Well drained")).strip()
        hydric  = str(attrs.get("HYDCLPRS",   "N")).strip()

        # Map SSURGO drainage class to moisture proxy
        drain_moisture = {
            "Excessively drained":    0.05,
            "Somewhat excessively drained": 0.12,
            "Well drained":           0.25,
            "Moderately well drained":0.40,
            "Somewhat poorly drained":0.55,
            "Poorly drained":         0.70,
            "Very poorly drained":    0.85,
        }
        moisture = drain_moisture.get(drain, 0.35)

        self._log(f"Soil → {name} | Drainage: {drain} | Hydric: {hydric} | Moisture: {moisture:.0%}")
        return {"name": name, "drainage": drain, "hydric": hydric, "moisture": moisture}

    def query_water_nearby(self, lat: float, lon: float, buffer_m: float = 500) -> bool:
        """Check for lakes/ponds/wetlands within buffer using NHD Water Bodies layer."""
        self._log(f"Checking water bodies within {buffer_m:.0f}m (NHD)...")
        ux, uy = latlon_to_utm16n(lat, lon)
        result = self._identify(self.WATER_SVC, ux, uy, pad=buffer_m)
        found = result is not None
        self._log(f"Water body found: {'YES — reduces spread' if found else 'No'}")
        return found

    def query_elevation(self, lat: float, lon: float) -> Optional[float]:
        """USGS Elevation Point Query Service — returns metres."""
        self._log("Querying elevation (USGS EPQS)...")
        try:
            r = requests.get(self.USGS_ELEV,
                             params={"x": lon, "y": lat, "units": "Meters", "output": "json"},
                             timeout=self.TIMEOUT)
            elev = float(r.json()["USGS_Elevation_Point_Query_Service"]
                                  ["Elevation_Query"]["Elevation"])
            self._log(f"Elevation: {elev:.1f} m ASL")
            return elev
        except Exception as e:
            self._log(f"Elevation query failed: {e}")
            return None

    def build_terrain_profile(self, lat: float, lon: float) -> TerrainProfile:
        """
        Compose a full TerrainProfile from multiple IndianaMap queries.
        Falls back gracefully if any service is unavailable.
        """
        print(f"\n  ┌─── IndianaMap Terrain Fetch ─── ({lat:.4f}°N, {abs(lon):.4f}°W) ──────────")

        nlcd      = self.query_land_cover(lat, lon)
        fuel_data = NLCD_FUEL_MODELS.get(nlcd, NLCD_FUEL_MODELS[DEFAULT_NLCD])
        soil      = self.query_soils(lat, lon)
        water     = self.query_water_nearby(lat, lon)

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
        print(f"  │")
        print(f"  │  ● Land Cover  : {profile.land_cover_name}  (NLCD {nlcd})")
        print(f"  │  ● Base Mult   : {profile.fuel_mult:.2f}  →  Effective: {profile.effective_mult():.2f}")
        print(f"  │  ● Soil        : {profile.soil_name[:40]}")
        print(f"  │  ● Drainage    : {profile.drainage_class}  (moisture {profile.soil_moisture:.0%})")
        print(f"  │  ● Hydric Soil : {profile.hydric_rating}  (wetland penalty applied if Y)")
        print(f"  │  ● Water Nearby: {'YES' if water else 'No'}")
        print(f"  │  ● Risk Level  : {profile.risk_level.upper()}")
        print(f"  └────────────────────────────────────────────────────────────────────\n")
        return profile


# ─────────────────────────────────────────────────────────────────────────────
# 5-STATE EXTENDED KALMAN FILTER  [cx, cy, a, b, θ]
#   cx, cy  — fire centroid (m)
#   a       — semi-major axis (head-fire direction, m)
#   b       — semi-minor axis (flanking fire, m)
#   θ       — ellipse orientation (radians, = wind direction)
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── accessors ──────────────────────────────────────────────────────────
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

    # ── Kalman predict step ─────────────────────────────────────────────────
    def predict(self, dx: float, dy: float, da: float, db: float, theta: float):
        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0]  = max(self.x[2, 0] + da, 0.5)
        self.x[3, 0]  = max(self.x[3, 0] + db, 0.5)
        self.x[4, 0]  = theta
        self.P        = self.P + self.Q

    # ── Kalman update step ──────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# ROTHERMEL + ANDREWS FIRE SPREAD MODEL
# Computes elliptical spread deltas for the KF predict step.
# Ref: Andrews (1986) — length-to-breadth ratio from wind speed.
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# RISK ASSESSOR
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# FERDA v2.0 — MAIN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── FIGURE ───────────────────────────────────────────────────────────────
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

        # ── Map panel ────────────────────────────────────────────────────────
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
            f"🌱 {self.terrain.land_cover_name}  ·  Soil: {self.terrain.soil_name[:28]}\n"
            f"Fuel ×{self.terrain.fuel_mult:.1f}  ·  Effective ×{self.terrain.effective_mult():.2f}"
            f"  ·  Moisture {self.terrain.soil_moisture:.0%}"
            + ("  ·  ⚠ Water barrier" if self.terrain.has_water_nearby else ""),
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

        # ── Growth chart ─────────────────────────────────────────────────────
        self.ax_growth.set_title("Fire Axes (m)", color="#ccccee", fontsize=8, pad=3)
        self.ax_growth.set_xlabel("Time (s)", fontsize=7)
        self.ax_growth.grid(True, alpha=0.18, color="#33335a")
        self.ln_major, = self.ax_growth.plot([], [], color="#FF4500", lw=1.6, label="Semi-major")
        self.ln_minor, = self.ax_growth.plot([], [], color="#FFD700", lw=1.2,
                                              linestyle="--", label="Semi-minor")
        self.ax_growth.legend(facecolor="#0d0d1f", edgecolor="#33335a",
                               labelcolor="#ccccee", fontsize=6.5)

        # ── Area chart ───────────────────────────────────────────────────────
        self.ax_area.set_title("Fire Area (ha)", color="#ccccee", fontsize=8, pad=3)
        self.ax_area.set_xlabel("Time (s)", fontsize=7)
        self.ax_area.grid(True, alpha=0.18, color="#33335a")
        self.ln_area, = self.ax_area.plot([], [], color="#9B00FF", lw=1.5)

        # ── Alerts panel ──────────────────────────────────────────────────────
        self.ax_alerts.axis("off")
        self.ax_alerts.set_title("System Alerts & State", color="#ccccee",
                                  fontsize=8.5, pad=6)
        self.txt_alerts = self.ax_alerts.text(
            0.04, 0.97, "", transform=self.ax_alerts.transAxes,
            fontsize=6.8, va="top", color="#ccccee", fontfamily="monospace",
            linespacing=1.55,
        )

    # ── PREDICT ──────────────────────────────────────────────────────────────
    def predict(self, dt: float, wind_speed: float, wind_dir: float, slope: float):
        self.wind_speed = wind_speed
        self.wind_dir   = wind_dir
        dx, dy, da, db, theta = self.spread.compute_increments(
            wind_speed, wind_dir, slope, dt)
        self.kf.predict(dx, dy, da, db, theta)
        self.current_time += dt
        self._record()

    # ── UPDATE ───────────────────────────────────────────────────────────────
    def update(self, obs: DroneObservation):
        obs.timestamp = self.current_time
        self.observations.append(obs)
        self.kf.update(obs)

    # ── RENDER ───────────────────────────────────────────────────────────────
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
            f"⏱  {self.current_time:.0f} s\n"
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

    # ── HELPERS ──────────────────────────────────────────────────────────────
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
        print(f"  ✓ Log exported → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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
    print("\n  ┌─ Runtime Commands ───────────────────────────────────────────┐")
    print("  │  <Enter>              → auto-step (dt=10 s)                   │")
    print("  │  x,y,r               → fuse drone observation (Drone 0)       │")
    print("  │  x,y,r,id            → multi-drone (Drone id)                 │")
    print("  │  x,y,r,id,conf       → with confidence 0–1                    │")
    print("  │  env                  → reconfigure wind / slope              │")
    print("  │  export               → write CSV log                         │")
    print("  │  quit / Ctrl-C        → exit                                  │")
    print("  └──────────────────────────────────────────────────────────────┘\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_banner()

    # ── Step 1: Location ─────────────────────────────────────────────────────
    print("\n  [1/3]  Incident location (Indiana WGS84 coordinates)")
    print("         Default: West Lafayette / Tippecanoe County")
    lat = get_float("  Latitude  [40.4259] : ", default=40.4259)
    lon = get_float("  Longitude [-86.9081]: ", default=-86.9081)

    # ── Step 2: IndianaMap terrain fetch ─────────────────────────────────────
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

    # ── Step 3: Fire initialisation ──────────────────────────────────────────
    print("\n  [3/3]  Initial fire parameters")
    ix = get_float("  Initial X (m) [0]   : ", default=0.0)
    iy = get_float("  Initial Y (m) [0]   : ", default=0.0)
    ir = get_float("  Initial Radius (m) [5]: ", default=5.0)

    ferda = FERDA_System(ix, iy, ir, terrain)
    print_controls()

    try:
        while True:
            # ── Environment config ────────────────────────────────────────────
            print("\n  ─── Environment ─────────────────────────────────────────────")
            wind_v = get_float("  Wind Speed (m/s): ")
            wind_d = get_float("  Wind Direction (° from East, CCW): ")
            slope  = get_float("  Slope Angle (°): ")

            # ── Simulation loop ───────────────────────────────────────────────
            print("\n  ─── Simulation running (type 'env' to reconfigure) ──────────")
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
                    print(f"  ✓ Observation from Drone {obs.drone_id} fused "
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
