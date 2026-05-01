from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

import firebase_admin
from firebase_admin import credentials, firestore


# ──────────────────────────────────────────────────────────────────────────────
#  Logger
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Elite")


# ──────────────────────────────────────────────────────────────────────────────
#  Firebase
# ──────────────────────────────────────────────────────────────────────────────

_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if _creds_json:
    try:
        json.loads(_creds_json)
    except json.JSONDecodeError as e:
        print(f"❌ GOOGLE_APPLICATION_CREDENTIALS_JSON مش JSON صالح → {e}", file=sys.stderr)
        sys.exit(1)
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    _tmp.write(_creds_json)
    _tmp.close()
    _SERVICE_KEY = _tmp.name
    print("✅ Firebase: Environment Variable")
else:
    _SERVICE_KEY = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
    if not os.path.exists(_SERVICE_KEY):
        print("❌ serviceAccountKey.json مش موجود", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Firebase: ملف محلي → {_SERVICE_KEY}")

if not firebase_admin._apps:
    cred = credentials.Certificate(_SERVICE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ──────────────────────────────────────────────────────────────────────────────
#  إعدادات التسعير
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PricingConfig:
    base_fare_egp: float = 15.0
    per_km_fare_egp: float = 8.0
    per_minute_fare_egp: float = 0.75
    minimum_fare_egp: float = 25.0
    fuel_price_per_liter_egp: float = 14.0
    fuel_consumption_per_100km: float = 9.5
    fuel_cost_coverage_ratio: float = 0.65
    surge_thresholds: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)
    surge_multipliers: tuple = (0.90, 1.00, 1.20, 1.50, 1.80, 2.20)
    late_night_bonus: float = 0.10
    rush_hour_bonus: float = 0.05
    service_fee_egp: float = 3.0
    platform_commission: float = 0.20


PRICING = PricingConfig()


# ──────────────────────────────────────────────────────────────────────────────
#  Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lon: float = Field(..., ge=-180, le=180)
    dest_lat: float = Field(..., ge=-90, le=90)
    dest_lon: float = Field(..., ge=-180, le=180)
    demand_factor: float = Field(default=1.0, ge=0.1, le=5.0)


class PriceEstimateRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lon: float = Field(..., ge=-180, le=180)
    dest_lat: float = Field(..., ge=-90, le=90)
    dest_lon: float = Field(..., ge=-180, le=180)
    demand_factor: float = Field(default=1.0, ge=0.1, le=5.0)
    trip_id: str | None = None


class GeocodeRequest(BaseModel):
    address: str = Field(..., min_length=3, max_length=300)

    @field_validator("address")
    @classmethod
    def strip_address(cls, v: str) -> str:
        return v.strip()


class ReverseGeocodeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class BatchGeocodeRequest(BaseModel):
    addresses: list[str] = Field(..., min_length=1, max_length=20)


class MatchRequest(BaseModel):
    user_id: str
    user_lat: float
    user_lon: float
    destination: str
    trip_id: str | None = None
    max_distance_km: float = Field(default=15.0, ge=1.0, le=50.0)


class TripStatusRequest(BaseModel):
    trip_id: str
    status: str
    driver_id: str


class DriverProfileRequest(BaseModel):
    driver_id: str
    name: str
    phone: str
    car_brand: str
    car_model: str
    car_year: int
    car_color: str
    plate_number: str
    national_id: str
    profile_photo_url: str | None = None
    car_photo_url: str | None = None


class UserProfileRequest(BaseModel):
    user_id: str
    name: str
    phone: str
    profile_photo_url: str | None = None


class RatingRequest(BaseModel):
    trip_id: str
    driver_id: str
    rating: float = Field(ge=1.0, le=5.0)
    comment: str | None = None


class SurgePricingRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 3.0


class MapRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    driver_lat: float | None = None
    driver_lon: float | None = None
    trip_id: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
#  Cache
# ──────────────────────────────────────────────────────────────────────────────

class _GeoCache:
    _routes: TTLCache = TTLCache(maxsize=500, ttl=600)
    _geocodes: TTLCache = TTLCache(maxsize=1000, ttl=3600)

    @classmethod
    def route_key(cls, olat, olon, dlat, dlon):
        return f"{round(olat,4)},{round(olon,4)}-{round(dlat,4)},{round(dlon,4)}"

    @classmethod
    def geocode_key(cls, text):
        return text.lower().strip()

    @classmethod
    def get_route(cls, key):
        return cls._routes.get(key)

    @classmethod
    def set_route(cls, key, value):
        cls._routes[key] = value

    @classmethod
    def get_geocode(cls, key):
        return cls._geocodes.get(key)

    @classmethod
    def set_geocode(cls, key, value):
        cls._geocodes[key] = value


GeoCache = _GeoCache()


# ──────────────────────────────────────────────────────────────────────────────
#  Geo Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _eta_minutes(dist_km, speed_kmh=20.0):
    return max(1, round(dist_km / (speed_kmh / 60)))


def _cairo_hour():
    cairo_tz = timezone(timedelta(hours=2))
    return datetime.now(cairo_tz).hour


def _is_rush_hour():
    h = _cairo_hour()
    return any(s <= h < e for s, e in [(7, 10), (16, 20)])


def _is_late_night():
    h = _cairo_hour()
    return h >= 23 or h < 5


# ──────────────────────────────────────────────────────────────────────────────
#  Route Engine (OSRM أولاً ← مجاني بالكامل، ثم Haversine كـ fallback)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    distance_km: float
    duration_min: float
    geometry: list[list[float]]
    source: str


class RouteEngine:
    _OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"
    _TIMEOUT = 5.0

    @classmethod
    async def get_route(cls, origin_lat, origin_lon, dest_lat, dest_lon) -> RouteResult:
        cache_key = GeoCache.route_key(origin_lat, origin_lon, dest_lat, dest_lon)
        cached = GeoCache.get_route(cache_key)
        if cached:
            return RouteResult(**cached)

        try:
            result = await cls._fetch_osrm(origin_lat, origin_lon, dest_lat, dest_lon)
        except Exception as e:
            log.warning(f"OSRM فشل، استخدام Haversine: {e}")
            result = cls._haversine_fallback(origin_lat, origin_lon, dest_lat, dest_lon)

        GeoCache.set_route(cache_key, {
            "distance_km": result.distance_km,
            "duration_min": result.duration_min,
            "geometry": result.geometry,
            "source": result.source,
        })
        return result

    @classmethod
    async def _fetch_osrm(cls, olat, olon, dlat, dlon) -> RouteResult:
        url = f"{cls._OSRM_BASE}/{olon},{olat};{dlon},{dlat}"
        params = {"overview": "simplified", "geometries": "geojson", "steps": "false"}
        async with httpx.AsyncClient(timeout=cls._TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            raise ValueError("OSRM: لم يُرجع مسار صالح")

        route = data["routes"][0]
        return RouteResult(
            distance_km=round(route["distance"] / 1000, 2),
            duration_min=round(route["duration"] / 60, 1),
            geometry=route["geometry"]["coordinates"],
            source="osrm",
        )

    @staticmethod
    def _haversine_fallback(lat1, lon1, lat2, lon2) -> RouteResult:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        straight = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist_km = round(straight * 1.35, 2)
        return RouteResult(
            distance_km=dist_km,
            duration_min=round((dist_km / 25.0) * 60, 1),
            geometry=[[lon1, lat1], [lon2, lat2]],
            source="haversine_fallback",
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Geocoding Engine (Nominatim - مجاني بالكامل)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeocodeResult:
    lat: float
    lon: float
    display_name: str
    place_type: str
    confidence: float


class GeocodingEngine:
    _NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
    _HEADERS = {"User-Agent": "Elite-App/2.2 (Elite@example.com)"}
    _TIMEOUT = 6.0
    _EGYPT_VIEWBOX = "24.70,21.97,36.90,31.67"

    @classmethod
    async def geocode(cls, address: str) -> GeocodeResult:
        key = GeoCache.geocode_key(f"fwd:{address}")
        cached = GeoCache.get_geocode(key)
        if cached:
            return GeocodeResult(**cached)
        result = await cls._nominatim_forward(address)
        GeoCache.set_geocode(key, result.__dict__)
        return result

    @classmethod
    async def reverse_geocode(cls, lat: float, lon: float) -> GeocodeResult:
        key = GeoCache.geocode_key(f"rev:{round(lat,5)},{round(lon,5)}")
        cached = GeoCache.get_geocode(key)
        if cached:
            return GeocodeResult(**cached)
        result = await cls._nominatim_reverse(lat, lon)
        GeoCache.set_geocode(key, result.__dict__)
        return result

    @classmethod
    async def _nominatim_forward(cls, address: str) -> GeocodeResult:
        params = {
            "q": address, "format": "jsonv2", "limit": 1,
            "viewbox": cls._EGYPT_VIEWBOX, "bounded": 1, "addressdetails": 1,
        }
        try:
            async with httpx.AsyncClient(headers=cls._HEADERS, timeout=cls._TIMEOUT) as client:
                resp = await client.get(f"{cls._NOMINATIM_BASE}/search", params=params)
                resp.raise_for_status()
                results = resp.json()

            if not results:
                params.pop("viewbox"), params.pop("bounded")
                async with httpx.AsyncClient(headers=cls._HEADERS, timeout=cls._TIMEOUT) as client:
                    resp = await client.get(f"{cls._NOMINATIM_BASE}/search", params=params)
                    resp.raise_for_status()
                    results = resp.json()

            if not results:
                raise HTTPException(status_code=404, detail="لم يتم العثور على العنوان")

            r = results[0]
            return GeocodeResult(
                lat=float(r["lat"]), lon=float(r["lon"]),
                display_name=r.get("display_name", ""),
                place_type=r.get("type", "unknown"),
                confidence=min(float(r.get("importance", 0.5)), 1.0),
            )
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"خدمة الخرائط غير متاحة: {e}")

    @classmethod
    async def _nominatim_reverse(cls, lat: float, lon: float) -> GeocodeResult:
        params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 16}
        try:
            async with httpx.AsyncClient(headers=cls._HEADERS, timeout=cls._TIMEOUT) as client:
                resp = await client.get(f"{cls._NOMINATIM_BASE}/reverse", params=params)
                resp.raise_for_status()
                r = resp.json()

            if "error" in r:
                raise HTTPException(status_code=404, detail="لا يوجد عنوان لهذه الإحداثيات")

            return GeocodeResult(
                lat=lat, lon=lon,
                display_name=r.get("display_name", ""),
                place_type=r.get("type", "unknown"),
                confidence=0.95,
            )
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
#  Trip Pricing Engine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceBreakdown:
    base_fare: float
    distance_cost: float
    time_cost: float
    fuel_component: float
    service_fee: float
    subtotal: float
    surge_multiplier: float
    surge_reason: str
    final_price: float
    driver_earnings: float
    is_surge: bool
    distance_km: float
    duration_min: float
    source: str


class TripPricingEngine:
    @classmethod
    def calculate(cls, route: RouteResult, demand_factor: float, is_rush=False, is_night=False) -> PriceBreakdown:
        cfg = PRICING
        base_fare = cfg.base_fare_egp
        distance_cost = round(route.distance_km * cfg.per_km_fare_egp, 2)
        time_cost = round(route.duration_min * cfg.per_minute_fare_egp, 2)
        service_fee = cfg.service_fee_egp
        fuel_component = round(
            (route.distance_km / 100) * cfg.fuel_consumption_per_100km
            * cfg.fuel_price_per_liter_egp * cfg.fuel_cost_coverage_ratio, 2
        )
        subtotal = round(base_fare + distance_cost + time_cost + fuel_component + service_fee, 2)

        multiplier, surge_reason = cls._compute_surge(demand_factor, is_rush, is_night)
        final_price = max(round(subtotal * multiplier / 5) * 5, cfg.minimum_fare_egp)
        driver_earnings = round(final_price * (1 - cfg.platform_commission), 1)

        return PriceBreakdown(
            base_fare=base_fare, distance_cost=distance_cost, time_cost=time_cost,
            fuel_component=fuel_component, service_fee=service_fee, subtotal=subtotal,
            surge_multiplier=round(multiplier, 2), surge_reason=surge_reason,
            final_price=float(final_price), driver_earnings=driver_earnings,
            is_surge=multiplier > 1.0, distance_km=route.distance_km,
            duration_min=route.duration_min, source=route.source,
        )

    @staticmethod
    def _compute_surge(demand_factor, is_rush, is_night):
        cfg = PRICING
        multiplier = cfg.surge_multipliers[0]
        for i, threshold in enumerate(cfg.surge_thresholds):
            if demand_factor > threshold:
                multiplier = cfg.surge_multipliers[i + 1]

        reasons = []
        if multiplier > 1.0:
            reasons.append(f"ضغط الطلب ({demand_factor:.1f}x)")
        elif multiplier < 1.0:
            reasons.append("عرض مرتفع — خصم تلقائي")
        if is_night:
            multiplier = round(multiplier + cfg.late_night_bonus, 2)
            reasons.append("رحلة ليلية")
        if is_rush:
            multiplier = round(multiplier + cfg.rush_hour_bonus, 2)
            reasons.append("ساعة الذروة")

        return multiplier, " + ".join(reasons) if reasons else "سعر عادي"

    @classmethod
    def to_firestore_dict(cls, b: PriceBreakdown) -> dict:
        return {
            "estimated_price": b.final_price,
            "price_breakdown": {
                "base_fare": b.base_fare, "distance_cost": b.distance_cost,
                "time_cost": b.time_cost, "fuel_component": b.fuel_component,
                "service_fee": b.service_fee, "subtotal": b.subtotal,
            },
            "surge_multiplier": b.surge_multiplier, "surge_reason": b.surge_reason,
            "is_surge": b.is_surge, "driver_earnings": b.driver_earnings,
            "distance_km": b.distance_km, "duration_min": b.duration_min,
            "route_source": b.source,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Context Analyser
# ──────────────────────────────────────────────────────────────────────────────

class ContextAnalyser:
    @staticmethod
    def now_cairo():
        return datetime.now(timezone(timedelta(hours=2)))

    @staticmethod
    def is_rush_hour():
        return _is_rush_hour()

    @staticmethod
    def is_late_night():
        return _is_late_night()

    @classmethod
    def get_demand_factor(cls, user_lat, user_lon, radius_km=3.0):
        try:
            trip_docs = db.collection("trips").where("status", "in", ["searching", "matched"]).limit(50).stream()
            demand = sum(
                1 for doc in trip_docs
                if _haversine(user_lat, user_lon, doc.to_dict().get("user_lat", 0), doc.to_dict().get("user_lon", 0)) <= radius_km
            )
            driver_docs = db.collection("users").where("role", "==", "driver").where("is_available", "==", True).limit(50).stream()
            supply = sum(
                1 for doc in driver_docs
                if _haversine(user_lat, user_lon, doc.to_dict().get("lat", 0), doc.to_dict().get("lon", 0)) <= radius_km
            )
            if supply == 0:
                return 3.0
            return round(demand / supply, 2)
        except Exception:
            return 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  Reliability & Responsiveness Scorers
# ──────────────────────────────────────────────────────────────────────────────

class ReliabilityScorer:
    @staticmethod
    def get_score(driver: dict) -> float:
        if time.time() < driver.get("blacklist_until", 0):
            return 0.0
        accepted = int(driver.get("total_accepted", 0))
        rejected = int(driver.get("total_rejected", 0))
        cancelled = int(driver.get("total_cancelled", 0))
        total_seen = accepted + rejected
        acceptance_rate = accepted / total_seen if total_seen else 0.85
        cancellation_rate = cancelled / accepted if accepted else 0.0
        return round((acceptance_rate * 0.6) + ((1 - min(cancellation_rate, 1.0)) * 0.4), 4)


class ResponsivenessScorer:
    @staticmethod
    def get_score(driver: dict) -> float:
        avg = float(driver.get("avg_response_seconds", 45.0))
        if avg <= 30: return 1.0
        elif avg <= 60: return 0.85
        elif avg <= 90: return 0.65
        elif avg <= 150: return 0.40
        else: return 0.20


# ──────────────────────────────────────────────────────────────────────────────
#  Blacklist Manager
# ──────────────────────────────────────────────────────────────────────────────

class BlacklistManager:
    @staticmethod
    def record_rejection(driver_id: str):
        ref = db.collection("users").document(driver_id)
        doc = ref.get()
        if not doc.exists:
            return
        data = doc.to_dict()
        consecutive = int(data.get("consecutive_rejects", 0)) + 1
        updates = {
            "consecutive_rejects": consecutive,
            "total_rejected": firestore.Increment(1),
        }
        if consecutive >= 3:
            updates["blacklist_until"] = time.time() + 15 * 60
            updates["consecutive_rejects"] = 0
            log.warning(f"Driver {driver_id} blacklisted 15 min")
        ref.update(updates)

    @staticmethod
    def record_acceptance(driver_id: str):
        db.collection("users").document(driver_id).update({
            "consecutive_rejects": 0,
            "total_accepted": firestore.Increment(1),
        })


# ──────────────────────────────────────────────────────────────────────────────
#  Surge Pricing Engine
# ──────────────────────────────────────────────────────────────────────────────

class SurgePricingEngine:
    BASE_FARE = 15.0
    PER_KM_FARE = 8.0

    @classmethod
    def calculate_price(cls, dist_km, demand_factor) -> dict:
        base_price = cls.BASE_FARE + dist_km * cls.PER_KM_FARE
        if demand_factor <= 0.5: multiplier = 0.9
        elif demand_factor <= 1.0: multiplier = 1.0
        elif demand_factor <= 1.5: multiplier = 1.2
        elif demand_factor <= 2.0: multiplier = 1.5
        elif demand_factor <= 3.0: multiplier = 1.8
        else: multiplier = 2.2

        if _is_late_night():
            multiplier += 0.1

        return {
            "base_price": round(base_price, 0),
            "multiplier": round(multiplier, 2),
            "final_price": round(base_price * multiplier, 0),
            "demand_factor": demand_factor,
            "is_surge": multiplier > 1.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LLM DRIVER AGENT  —  بديل كامل لـ UtilityAgent
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_LLM_URL    = "https://api.anthropic.com/v1/messages"
_LLM_MODEL  = "claude-haiku-4-5-20251001"
_LLM_TOKENS = 200

_SELECT_DRIVER_TOOL = {
    "name": "select_driver",
    "description": "Select the single best driver from the candidates list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_driver_id": {
                "type": "string",
                "description": "The _id of the chosen driver",
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences max explaining the choice",
            },
        },
        "required": ["selected_driver_id", "reason"],
    },
}

_SYSTEM_PROMPT = (
    "You are a ride-hailing dispatcher. "
    "Given a list of available drivers and trip context, call select_driver with the best choice. "
    "Priority order: lowest ETA → highest rating → highest reliability. "
    "Never pick a blacklisted driver. Keep reason under 2 sentences."
)


def _build_llm_prompt(candidates: list[dict], context: dict) -> str:
    ctx = (
        f"rush_hour={context['rush_hour']},"
        f"late_night={context['late_night']},"
        f"demand={context['demand_factor']}"
    )
    rows = "\n".join(
        f"id={c['id']},dist={c['dist_km']}km,eta={c['eta_min']}min,"
        f"rating={c['rating']},reliability={c['reliability']},response={c['response_score']}"
        for c in candidates
    )
    return f"context:{ctx}\ncandidates:\n{rows}"


async def _call_llm(candidates: list[dict], context: dict) -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None

    payload = {
        "model": _LLM_MODEL,
        "max_tokens": _LLM_TOKENS,
        "system": _SYSTEM_PROMPT,
        "tools": [_SELECT_DRIVER_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": _build_llm_prompt(candidates, context)}],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(_LLM_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "select_driver":
                return block["input"]
        return None
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return None


def _fallback_score(c: dict) -> float:
    """
    Backup scorer that mirrors the old UtilityAgent weights when LLM unavailable:
    proximity(40%) + quality(25%) + reliability(20%) + response(15%)
    """
    eta_score    = max(0.0, 1.0 - c["eta_min"] / 30.0)
    rating_score = (c["rating"] - 1.0) / 4.0
    return (
        0.40 * eta_score
        + 0.25 * rating_score
        + 0.20 * c["reliability"]
        + 0.15 * c["response_score"]
    )


class LLMDriverAgent:
    """
    Exact drop-in replacement for the old UtilityAgent.
    Applies the same pre-filtering & exclusion logic, then delegates
    the final selection to the LLM (or fallback scoring if LLM unavailable).
    """
    MAX_CANDIDATES = 5
    MAX_DIST_KM    = 20.0

    @classmethod
    def _pre_filter_and_enrich(cls, drivers: list[dict], user_lat: float, user_lon: float) -> list[dict]:
        """
        Same exclusion rules as old UtilityAgent.compute_utility:
          • Skip blacklisted drivers
          • Skip drivers beyond MAX_DIST_KM
          • Skip drivers where reliability == 0
        Enriches each surviving driver with computed fields.
        Sorted by ETA asc → top MAX_CANDIDATES forwarded to LLM.
        """
        now = time.time()
        enriched = []
        for d in drivers:
            if now < d.get("blacklist_until", 0):
                continue

            dist = _haversine(
                user_lat, user_lon,
                float(d.get("lat", user_lat)),
                float(d.get("lon", user_lon)),
            )
            if dist > cls.MAX_DIST_KM:
                continue

            reliability = ReliabilityScorer.get_score(d)
            if reliability == 0.0:
                continue

            enriched.append({
                "id":             d["_id"],
                "dist_km":        round(dist, 2),
                "eta_min":        _eta_minutes(dist),
                "rating":         float(d.get("rating", 4.5)),
                "reliability":    reliability,
                "response_score": ResponsivenessScorer.get_score(d),
                "_raw":           d,
            })

        enriched.sort(key=lambda x: x["eta_min"])
        return enriched[: cls.MAX_CANDIDATES]

    @classmethod
    async def select_best(
        cls,
        drivers: list[dict],
        user_lat: float,
        user_lon: float,
        demand_factor: float,
    ) -> tuple[dict, dict]:
        """
        Returns (best_candidate, agent_meta).
        """
        candidates = cls._pre_filter_and_enrich(drivers, user_lat, user_lon)

        if not candidates:
            raise HTTPException(status_code=404, detail="no_drivers_available")

        context = {
            "rush_hour":     _is_rush_hour(),
            "late_night":    _is_late_night(),
            "demand_factor": demand_factor,
        }

        llm_result = await _call_llm(candidates, context)

        if llm_result:
            chosen_id = llm_result.get("selected_driver_id")
            matched = next((c for c in candidates if c["id"] == chosen_id), None)
            if matched:
                return matched, {
                    "method":               "llm_agent",
                    "model":                _LLM_MODEL,
                    "reason":               llm_result.get("reason", ""),
                    "candidates_evaluated": len(candidates),
                }
            log.warning(f"LLM returned unknown driver_id={chosen_id}, falling back")

        best = max(candidates, key=_fallback_score)
        return best, {
            "method":               "fallback_scoring",
            "model":                None,
            "reason":               (
                f"ETA {best['eta_min']}min, "
                f"rating {best['rating']}, "
                f"reliability {round(best['reliability'], 2)}"
            ),
            "candidates_evaluated": len(candidates),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Map Generator (OpenStreetMap + Leaflet - مجاني 100%)
# ──────────────────────────────────────────────────────────────────────────────

def generate_trip_map(origin_lat, origin_lon, dest_lat, dest_lon, route_coords=None, driver_lat=None, driver_lon=None) -> str:
    if route_coords and len(route_coords) > 1:
        route_js = str([[c[1], c[0]] for c in route_coords])
    else:
        route_js = f"[[{origin_lat},{origin_lon}],[{dest_lat},{dest_lon}]]"

    center_lat = (origin_lat + dest_lat) / 2
    center_lon = (origin_lon + dest_lon) / 2

    driver_marker = ""
    if driver_lat and driver_lon:
        driver_marker = f"""
        var driverIcon = L.divIcon({{
            html: '<div style="font-size:24px;">🚗</div>',
            iconAnchor: [12, 12],
            className: ''
        }});
        L.marker([{driver_lat},{driver_lon}], {{icon: driverIcon}})
            .addTo(map).bindPopup('السائق هنا').openPopup();
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>خريطة الرحلة - Elite</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin: 0; padding: 0; font-family: Arial; }}
  #map {{ width: 100%; height: 100vh; }}
  .legend {{
    position: absolute; bottom: 20px; right: 10px;
    background: white; padding: 10px; border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 1000; font-size: 13px;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  📍 نقطة الانطلاق<br>
  🏁 الوجهة<br>
  🚗 السائق<br>
  ─── المسار
</div>
<script>
  var map = L.map('map').setView([{center_lat},{center_lon}], 13);

  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }}).addTo(map);

  var routeCoords = {route_js};
  L.polyline(routeCoords, {{color: '#2563eb', weight: 4, opacity: 0.8}}).addTo(map);

  var startIcon = L.divIcon({{
    html: '<div style="font-size:28px;">📍</div>',
    iconAnchor: [14, 28], className: ''
  }});
  L.marker([{origin_lat},{origin_lon}], {{icon: startIcon}})
    .addTo(map).bindPopup('نقطة الانطلاق');

  var endIcon = L.divIcon({{
    html: '<div style="font-size:28px;">🏁</div>',
    iconAnchor: [14, 28], className: ''
  }});
  L.marker([{dest_lat},{dest_lon}], {{icon: endIcon}})
    .addTo(map).bindPopup('الوجهة');

  {driver_marker}

  map.fitBounds(routeCoords);
</script>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Elite LLM Agent",
    version="3.0.0",
    description="LLM-powered driver selection — كل وظائف النظام الأصلي + LLM agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "PUT"],
    allow_headers=["Content-Type"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    return {
        "status": "ok",
        "service": "Elite LLM Agent",
        "version": "3.0.0",
        "agent_mode": "llm" if ANTHROPIC_API_KEY else "fallback_only",
        "llm_model": _LLM_MODEL,
        "time_cairo": ContextAnalyser.now_cairo().isoformat(),
        "is_rush_hour": ContextAnalyser.is_rush_hour(),
        "is_late_night": ContextAnalyser.is_late_night(),
    }


# ── Geo Routes ────────────────────────────────────────────────────────────────

@app.post("/geo/route", tags=["geo"])
async def get_route(req: RouteRequest):
    try:
        route = await RouteEngine.get_route(req.origin_lat, req.origin_lon, req.dest_lat, req.dest_lon)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"خطأ في حساب المسار: {e}")

    is_rush  = _is_rush_hour()
    is_night = _is_late_night()
    pricing  = TripPricingEngine.calculate(route, req.demand_factor, is_rush, is_night)

    return {
        "route": {
            "distance_km": route.distance_km,
            "duration_min": route.duration_min,
            "geometry": route.geometry,
            "source": route.source,
        },
        "pricing": {
            "final_price": pricing.final_price,
            "driver_earnings": pricing.driver_earnings,
            "is_surge": pricing.is_surge,
            "surge_multiplier": pricing.surge_multiplier,
            "surge_reason": pricing.surge_reason,
            "breakdown": {
                "base_fare": pricing.base_fare,
                "distance_cost": pricing.distance_cost,
                "time_cost": pricing.time_cost,
                "fuel_component": pricing.fuel_component,
                "service_fee": pricing.service_fee,
                "subtotal": pricing.subtotal,
            },
        },
    }


@app.post("/geo/price-estimate", tags=["geo"])
async def price_estimate(req: PriceEstimateRequest):
    try:
        route = await RouteEngine.get_route(req.origin_lat, req.origin_lon, req.dest_lat, req.dest_lon)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"خطأ مسار: {e}")

    pricing = TripPricingEngine.calculate(route, req.demand_factor, _is_rush_hour(), _is_late_night())

    if req.trip_id:
        try:
            updates = TripPricingEngine.to_firestore_dict(pricing)
            updates["price_updated_at"] = firestore.SERVER_TIMESTAMP
            db.collection("trips").document(req.trip_id).update(updates)
        except Exception as e:
            log.warning(f"Firestore update failed: {e}")

    return {
        "final_price": pricing.final_price,
        "driver_earnings": pricing.driver_earnings,
        "is_surge": pricing.is_surge,
        "surge_multiplier": pricing.surge_multiplier,
        "surge_reason": pricing.surge_reason,
        "distance_km": pricing.distance_km,
        "duration_min": pricing.duration_min,
        "breakdown": {
            "base_fare": pricing.base_fare,
            "distance_cost": pricing.distance_cost,
            "time_cost": pricing.time_cost,
            "fuel_component": pricing.fuel_component,
            "service_fee": pricing.service_fee,
            "subtotal": pricing.subtotal,
        },
        "route_source": pricing.source,
        "firestore_updated": bool(req.trip_id),
    }


@app.post("/geo/geocode", tags=["geo"])
async def geocode_address(req: GeocodeRequest):
    result = await GeocodingEngine.geocode(req.address)
    return {
        "lat": result.lat, "lon": result.lon,
        "display_name": result.display_name,
        "place_type": result.place_type,
        "confidence": result.confidence,
    }


@app.post("/geo/reverse-geocode", tags=["geo"])
async def reverse_geocode(req: ReverseGeocodeRequest):
    result = await GeocodingEngine.reverse_geocode(req.lat, req.lon)
    return {
        "display_name": result.display_name,
        "place_type": result.place_type,
        "lat": result.lat,
        "lon": result.lon,
    }


@app.post("/geo/geocode/batch", tags=["geo"])
async def batch_geocode(req: BatchGeocodeRequest):
    results = []
    for address in req.addresses:
        try:
            geo = await GeocodingEngine.geocode(address)
            results.append({
                "address": address, "lat": geo.lat, "lon": geo.lon,
                "display_name": geo.display_name, "confidence": geo.confidence, "status": "ok",
            })
        except HTTPException as e:
            results.append({"address": address, "status": "error", "detail": e.detail})
        await asyncio.sleep(0.25)
    return {"results": results, "count": len(results)}


@app.get("/geo/pricing-config", tags=["geo"])
def get_pricing_config():
    cfg = PRICING
    return {
        "base_fare_egp": cfg.base_fare_egp,
        "per_km_fare_egp": cfg.per_km_fare_egp,
        "per_minute_fare_egp": cfg.per_minute_fare_egp,
        "minimum_fare_egp": cfg.minimum_fare_egp,
        "service_fee_egp": cfg.service_fee_egp,
        "platform_commission_pct": cfg.platform_commission * 100,
        "fuel_price_per_liter_egp": cfg.fuel_price_per_liter_egp,
        "surge_late_night_bonus_pct": cfg.late_night_bonus * 100,
        "surge_rush_hour_bonus_pct": cfg.rush_hour_bonus * 100,
    }


@app.post("/geo/map", tags=["geo"])
async def get_trip_map(req: MapRequest):
    try:
        route  = await RouteEngine.get_route(req.origin_lat, req.origin_lon, req.dest_lat, req.dest_lon)
        coords = route.geometry
    except Exception:
        coords = None

    html = generate_trip_map(
        req.origin_lat, req.origin_lon,
        req.dest_lat, req.dest_lon,
        route_coords=coords,
        driver_lat=req.driver_lat,
        driver_lon=req.driver_lon,
    )
    return HTMLResponse(content=html)


@app.get("/geo/map", tags=["geo"])
async def get_map_simple(
    origin_lat: float = Query(...),
    origin_lon: float = Query(...),
    dest_lat:   float = Query(...),
    dest_lon:   float = Query(...),
    driver_lat: float | None = Query(default=None),
    driver_lon: float | None = Query(default=None),
):
    try:
        route  = await RouteEngine.get_route(origin_lat, origin_lon, dest_lat, dest_lon)
        coords = route.geometry
    except Exception:
        coords = None

    html = generate_trip_map(origin_lat, origin_lon, dest_lat, dest_lon, coords, driver_lat, driver_lon)
    return HTMLResponse(content=html)


# ── Match Driver — LLM Agent ──────────────────────────────────────────────────

@app.post("/match-driver", tags=["agent"])
async def match_driver(req: MatchRequest):
    try:
        docs = db.collection("users").where("role", "==", "driver").where("is_available", "==", True).stream()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Firestore error: {e}")

    drivers = []
    for doc in docs:
        d = doc.to_dict()
        d["_id"] = doc.id
        drivers.append(d)

    demand_factor = ContextAnalyser.get_demand_factor(req.user_lat, req.user_lon)

    # نفس منطق wait_seconds الأصلي
    wait_seconds = 0.0
    if req.trip_id:
        try:
            trip_doc = db.collection("trips").document(req.trip_id).get()
            if trip_doc.exists:
                created_at = trip_doc.to_dict().get("created_at")
                if created_at:
                    wait_seconds = time.time() - created_at.timestamp()
        except Exception:
            pass

    # LLM Agent يختار أفضل سائق بدل UtilityAgent
    best, agent_meta = await LLMDriverAgent.select_best(
        drivers, req.user_lat, req.user_lon, demand_factor
    )

    raw     = best["_raw"]
    dist_km = best["dist_km"]
    eta     = best["eta_min"]
    pricing = SurgePricingEngine.calculate_price(dist_km, demand_factor)

    if req.trip_id:
        try:
            db.collection("trips").document(req.trip_id).update({
                "driver_id":        best["id"],
                "driver_name":      raw.get("name", "سائق"),
                "driver_phone":     raw.get("phone", ""),
                "driver_lat":       raw.get("lat"),
                "driver_lon":       raw.get("lon"),
                "driver_rating":    raw.get("rating", 5.0),
                "driver_car":       f"{raw.get('car_brand','')} {raw.get('car_model','')}".strip() or raw.get("car", "سيارة"),
                "driver_car_color": raw.get("car_color", ""),
                "driver_plate":     raw.get("plate_number", raw.get("plate", "")),
                "driver_photo_url": raw.get("profile_photo_url", ""),
                "distance_km":      dist_km,
                "eta_min":          eta,
                "estimated_price":  pricing["final_price"],
                "surge_multiplier": pricing["multiplier"],
                "is_surge":         pricing["is_surge"],
                "status":           "matched",
                "matched_at":       firestore.SERVER_TIMESTAMP,
                "agent_method":     agent_meta["method"],
            })
        except Exception as e:
            log.warning(f"Firestore update error: {e}")

    return {
        "matched_driver": {
            "driver_id":   best["id"],
            "name":        raw.get("name", "سائق"),
            "phone":       raw.get("phone", ""),
            "rating":      raw.get("rating", 5.0),
            "car":         f"{raw.get('car_brand','')} {raw.get('car_model','')}".strip() or raw.get("car", "سيارة"),
            "car_color":   raw.get("car_color", ""),
            "plate":       raw.get("plate_number", raw.get("plate", "")),
            "photo_url":   raw.get("profile_photo_url", ""),
            "lat":         raw.get("lat"),
            "lon":         raw.get("lon"),
            "distance_km": dist_km,
            "eta_min":     eta,
        },
        "pricing": pricing,
        "agent_decision": {
            "method":               agent_meta["method"],
            "model":                agent_meta.get("model"),
            "reason":               agent_meta["reason"],
            "candidates_evaluated": agent_meta["candidates_evaluated"],
            "demand_factor":        demand_factor,
            "is_rush_hour":         ContextAnalyser.is_rush_hour(),
            "is_late_night":        ContextAnalyser.is_late_night(),
            "wait_seconds":         round(wait_seconds),
        },
    }


# ── Trip Status ───────────────────────────────────────────────────────────────

@app.post("/trip-status", tags=["trips"])
async def update_trip_status(req: TripStatusRequest):
    valid_statuses = {"accepted", "driver_on_way", "arrived", "in_progress", "completed", "cancelled"}
    if req.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"invalid_status. Valid: {valid_statuses}")

    ref = db.collection("trips").document(req.trip_id)
    trip = ref.get()
    if not trip.exists:
        raise HTTPException(status_code=404, detail="trip_not_found")

    trip_data = trip.to_dict()
    if trip_data.get("driver_id") != req.driver_id:
        raise HTTPException(status_code=403, detail="not_your_trip")

    driver_ref = db.collection("users").document(req.driver_id)
    updates = {"status": req.status, f"{req.status}_at": firestore.SERVER_TIMESTAMP}

    if req.status == "accepted":
        driver_ref.update({
            "is_available": False,
            "consecutive_rejects": 0,
            "total_accepted": firestore.Increment(1),
        })
    elif req.status == "in_progress":
        updates["trip_started_at"] = firestore.SERVER_TIMESTAMP
    elif req.status == "completed":
        price = float(trip_data.get("estimated_price", 0))
        driver_ref.update({
            "is_available": True,
            "trips_count": firestore.Increment(1),
            "trips_count_today": firestore.Increment(1),
            "total_earnings": firestore.Increment(price),
            "total_trips_ever": firestore.Increment(1),
        })
    elif req.status == "cancelled":
        driver_ref.update({"is_available": True})
        if trip_data.get("cancelled_by", "driver") == "driver":
            BlacklistManager.record_rejection(req.driver_id)

    ref.update(updates)
    return {"ok": True, "trip_id": req.trip_id, "status": req.status}


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.post("/driver/profile", tags=["profiles"])
async def save_driver_profile(req: DriverProfileRequest):
    ref = db.collection("users").document(req.driver_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    ref.update({
        "name": req.name, "phone": req.phone,
        "car_brand": req.car_brand, "car_model": req.car_model,
        "car_year": req.car_year, "car_color": req.car_color,
        "plate_number": req.plate_number, "national_id": req.national_id,
        "profile_photo_url": req.profile_photo_url or "",
        "car_photo_url": req.car_photo_url or "",
        "car": f"{req.car_brand} {req.car_model}",
        "plate": req.plate_number,
        "profile_complete": True,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    return {"ok": True, "message": "تم حفظ بيانات السائق"}


@app.post("/user/profile", tags=["profiles"])
async def save_user_profile(req: UserProfileRequest):
    ref = db.collection("users").document(req.user_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="user_not_found")

    ref.update({
        "name": req.name, "phone": req.phone,
        "profile_photo_url": req.profile_photo_url or "",
        "profile_complete": True,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    return {"ok": True, "message": "تم حفظ بيانات المستخدم"}


# ── Rating ────────────────────────────────────────────────────────────────────

@app.post("/trip/rate", tags=["trips"])
async def rate_driver(req: RatingRequest):
    driver_ref = db.collection("users").document(req.driver_id)
    doc = driver_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    data = doc.to_dict()
    current_rating = float(data.get("rating", 5.0))
    total_ratings  = int(data.get("total_ratings", 1))
    new_rating     = round((current_rating * total_ratings + req.rating) / (total_ratings + 1), 2)

    driver_ref.update({"rating": new_rating, "total_ratings": total_ratings + 1})
    db.collection("trips").document(req.trip_id).update({
        "user_rating":  req.rating,
        "user_comment": req.comment or "",
        "rated_at":     firestore.SERVER_TIMESTAMP,
    })
    return {"ok": True, "new_rating": new_rating, "total_ratings": total_ratings + 1}


# ── Driver Stats ──────────────────────────────────────────────────────────────

@app.get("/driver/{driver_id}/stats", tags=["analytics"])
async def driver_stats(driver_id: str):
    doc = db.collection("users").document(driver_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    data            = doc.to_dict()
    blacklist_until = data.get("blacklist_until", 0)

    return {
        "driver_id":        driver_id,
        "name":             data.get("name", "سائق"),
        "rating":           data.get("rating", 5.0),
        "trips_count":      data.get("trips_count", 0),
        "trips_count_today":data.get("trips_count_today", 0),
        "total_earnings":   data.get("total_earnings", 0.0),
        "is_available":     data.get("is_available", False),
        "profile_complete": data.get("profile_complete", False),
        "agent_scores": {
            "reliability":    ReliabilityScorer.get_score(data),
            "responsiveness": ResponsivenessScorer.get_score(data),
        },
        "blacklisted_seconds_remaining": max(0, round(blacklist_until - time.time())),
        "total_accepted":  data.get("total_accepted", 0),
        "total_rejected":  data.get("total_rejected", 0),
        "total_cancelled": data.get("total_cancelled", 0),
    }


# ── Surge Info ────────────────────────────────────────────────────────────────

@app.post("/surge-info", tags=["pricing"])
async def get_surge_info(req: SurgePricingRequest):
    demand_factor = ContextAnalyser.get_demand_factor(req.lat, req.lon, req.radius_km)
    pricing_5km   = SurgePricingEngine.calculate_price(5.0, demand_factor)
    pricing_10km  = SurgePricingEngine.calculate_price(10.0, demand_factor)

    return {
        "demand_factor":    demand_factor,
        "is_surge":         pricing_5km["is_surge"],
        "surge_multiplier": pricing_5km["multiplier"],
        "is_rush_hour":     ContextAnalyser.is_rush_hour(),
        "is_late_night":    ContextAnalyser.is_late_night(),
        "sample_prices":    {"5km": pricing_5km["final_price"], "10km": pricing_10km["final_price"]},
    }


# ── Area Analytics ────────────────────────────────────────────────────────────

@app.get("/analytics/area", tags=["analytics"])
async def area_analytics(
    lat:       float = Query(...),
    lon:       float = Query(...),
    radius_km: float = Query(default=5.0, ge=0.5, le=20.0),
):
    demand_factor = ContextAnalyser.get_demand_factor(lat, lon, radius_km)

    drivers_snap = db.collection("users").where("role", "==", "driver").where("is_available", "==", True).limit(100).stream()
    available_drivers = []
    for d in drivers_snap:
        data = d.to_dict()
        dlat, dlon = data.get("lat", 0), data.get("lon", 0)
        if _haversine(lat, lon, dlat, dlon) <= radius_km:
            available_drivers.append({
                "driver_id": d.id, "name": data.get("name", "سائق"),
                "rating": data.get("rating", 5.0), "lat": dlat, "lon": dlon,
            })

    return {
        "area_center":       {"lat": lat, "lon": lon},
        "radius_km":         radius_km,
        "available_drivers": len(available_drivers),
        "drivers_list":      available_drivers[:10],
        "demand_factor":     demand_factor,
        "is_rush_hour":      ContextAnalyser.is_rush_hour(),
        "is_late_night":     ContextAnalyser.is_late_night(),
        "surge_multiplier":  SurgePricingEngine.calculate_price(5.0, demand_factor)["multiplier"],
    }


# ── Cancel Trip ───────────────────────────────────────────────────────────────

@app.post("/trip/{trip_id}/cancel", tags=["trips"])
async def cancel_trip(trip_id: str, cancelled_by: str = Query(default="user")):
    ref  = db.collection("trips").document(trip_id)
    trip = ref.get()
    if not trip.exists:
        raise HTTPException(status_code=404, detail="trip_not_found")

    data = trip.to_dict()
    if data.get("status") in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail="trip_already_finished")

    ref.update({"status": "cancelled", "cancelled_by": cancelled_by, "cancelled_at": firestore.SERVER_TIMESTAMP})

    driver_id = data.get("driver_id")
    if driver_id:
        db.collection("users").document(driver_id).update({"is_available": True})
        if cancelled_by == "driver":
            BlacklistManager.record_rejection(driver_id)

    return {"ok": True, "trip_id": trip_id, "cancelled_by": cancelled_by}


# ── Driver Location ───────────────────────────────────────────────────────────

@app.put("/driver/{driver_id}/location", tags=["tracking"])
async def update_driver_location(
    driver_id: str,
    lat:       float = Query(...),
    lon:       float = Query(...),
    trip_id:   str | None = Query(default=None),
):
    db.collection("users").document(driver_id).update({
        "lat": lat, "lon": lon,
        "last_location_at": firestore.SERVER_TIMESTAMP,
    })
    if trip_id:
        try:
            db.collection("trips").document(trip_id).update({"driver_lat": lat, "driver_lon": lon})
        except Exception:
            pass
    return {"ok": True}
