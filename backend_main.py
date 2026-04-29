"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Harafy — Utility-Based AI Agent Backend  v2.0                 ║
║              FastAPI + Firebase + Advanced Multi-Criteria Agent             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │               PERCEPT  →  AGENT  →  ACTION              │
  │                                                         │
  │  Percepts:                                              │
  │    • Available drivers (location, rating, trips_today)  │
  │    • Time of day (rush hour?)                           │
  │    • Demand density in area                             │
  │    • Driver refusal history                             │
  │    • User wait time                                     │
  │                                                         │
  │  Utility Function (multi-criteria weighted):            │
  │    U = w1·proximity + w2·quality + w3·availability      │
  │      + w4·reliability + w5·responsiveness               │
  │                                                         │
  │  Weights are DYNAMIC — shift based on context:          │
  │    Rush hour → proximity weight ↑                       │
  │    High demand → availability weight ↑                  │
  │    Long wait → responsiveness weight ↑                  │
  │    Night → reliability weight ↑                         │
  │                                                         │
  │  Actions:                                               │
  │    • match_driver → update Firestore trip               │
  │    • update_trip_status → driver availability           │
  │    • surge_pricing → dynamic price multiplier           │
  │    • blacklist_temp → penalize serial refusers          │
  └─────────────────────────────────────────────────────────┘
"""
from geo_pricing_module import (
             RouteEngine, GeocodingEngine, TripPricingEngine,
             PriceEstimateRequest, GeocodeRequest, ReverseGeocodeRequest,
             RouteRequest,geo_router
         )

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import firebase_admin
from firebase_admin import credentials, firestore


# ══════════════════════════════════════════════════════════════════════════════
#  FIREBASE INIT
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Harafy Utility-Based Agent",
    version="2.0.0",
    description="AI Agent يختار أفضل سائق بناءً على utility function ديناميكية",
)

app.include_router(geo_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # في production: غيّر لـ domain بتاعك
    allow_methods=["POST", "GET", "PUT"],
    allow_headers=["Content-Type"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class MatchRequest(BaseModel):
    user_id: str
    user_lat: float
    user_lon: float
    destination: str
    trip_id: str | None = None
    max_distance_km: float = Field(default=15.0, ge=1.0, le=50.0)


class TripStatusRequest(BaseModel):
    trip_id: str
    status: str   # accepted | driver_on_way | arrived | in_progress | completed | cancelled
    driver_id: str


class DriverProfileRequest(BaseModel):
    """السائق يسجل معلوماته الكاملة"""
    driver_id: str
    name: str
    phone: str
    car_brand: str        # مثلاً: Toyota
    car_model: str        # مثلاً: Camry
    car_year: int
    car_color: str
    plate_number: str
    national_id: str
    profile_photo_url: str | None = None
    car_photo_url: str | None = None


class UserProfileRequest(BaseModel):
    """المستخدم يكمّل معلوماته"""
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


# ══════════════════════════════════════════════════════════════════════════════
#  GEO HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """المسافة بالكيلومتر بين نقطتين."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + (
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _eta_minutes(dist_km: float, speed_kmh: float = 20.0) -> int:
    return max(1, round(dist_km / (speed_kmh / 60)))


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT ANALYSER  — يقرأ الظروف الحالية
# ══════════════════════════════════════════════════════════════════════════════

class ContextAnalyser:
    """
    يحلل السياق الحالي ويرجع معاملات تؤثر على الـ weights.
    """

    RUSH_HOURS = [(7, 10), (16, 20)]   # صباحاً + مساءً
    LATE_NIGHT = (23, 5)               # ليلاً

    @staticmethod
    def now_cairo() -> datetime:
        """الوقت الحالي بتوقيت القاهرة (UTC+2 أو UTC+3 صيفاً)."""
        from datetime import timezone, timedelta
        cairo_tz = timezone(timedelta(hours=2))
        return datetime.now(cairo_tz)

    @classmethod
    def is_rush_hour(cls) -> bool:
        h = cls.now_cairo().hour
        return any(s <= h < e for s, e in cls.RUSH_HOURS)

    @classmethod
    def is_late_night(cls) -> bool:
        h = cls.now_cairo().hour
        return h >= cls.LATE_NIGHT[0] or h < cls.LATE_NIGHT[1]

    @classmethod
    def get_demand_factor(cls, user_lat: float, user_lon: float, radius_km: float = 3.0) -> float:
        """
        يحسب نسبة الطلب vs العرض في المنطقة.
        demand_factor > 1  →  طلب عالي → surge pricing
        demand_factor < 1  →  عرض عالي → سعر عادي
        """
        try:
            # عدد الرحلات الـ searching في آخر 10 دقائق في المنطقة
            # (تقريب بسيط بدون GeoHash — للإنتاج استخدم GeoFirestore)
            trip_docs = (
                db.collection("trips")
                .where("status", "in", ["searching", "matched"])
                .limit(50)
                .stream()
            )
            demand = 0
            for doc in trip_docs:
                d = doc.to_dict()
                lat = d.get("user_lat", 0)
                lon = d.get("user_lon", 0)
                if _haversine(user_lat, user_lon, lat, lon) <= radius_km:
                    demand += 1

            # عدد السائقين المتاحين في المنطقة
            driver_docs = (
                db.collection("users")
                .where("role", "==", "driver")
                .where("is_available", "==", True)
                .limit(50)
                .stream()
            )
            supply = 0
            for doc in driver_docs:
                d = doc.to_dict()
                lat = d.get("lat", 0)
                lon = d.get("lon", 0)
                if _haversine(user_lat, user_lon, lat, lon) <= radius_km:
                    supply += 1

            if supply == 0:
                return 3.0  # لا يوجد سائقين — أعلى ضغط
            return round(demand / supply, 2)
        except Exception:
            return 1.0

    @classmethod
    def compute_dynamic_weights(cls, demand_factor: float, wait_seconds: float = 0) -> dict:
        """
        الـ weights الديناميكية للـ utility function.
        
        Criteria:
          w_proximity    → قرب السائق من المستخدم
          w_quality      → تقييم السائق
          w_availability → نشاط السائق (مش محمّل بأوردرات كتير)
          w_reliability  → موثوقية السائق (مرفوضاتٍ قليلة)
          w_speed        → سرعة الاستجابة لو الطلب قديم
        """
        base = {
            "w_proximity":    0.40,
            "w_quality":      0.25,
            "w_availability": 0.15,
            "w_reliability":  0.12,
            "w_speed":        0.08,
        }

        # وقت الذروة → قرب أهم
        if cls.is_rush_hour():
            base["w_proximity"] += 0.08
            base["w_quality"] -= 0.05
            base["w_availability"] -= 0.03

        # ليل → موثوقية أهم (أمان)
        if cls.is_late_night():
            base["w_reliability"] += 0.08
            base["w_proximity"] -= 0.05
            base["w_quality"] += 0.02

        # طلب عالي → متاحية أهم
        if demand_factor > 1.5:
            base["w_availability"] += 0.10
            base["w_proximity"] -= 0.05
            base["w_quality"] -= 0.05

        # انتظار طويل → سرعة أهم
        if wait_seconds > 120:
            base["w_speed"] += 0.10
            base["w_quality"] -= 0.05
            base["w_availability"] -= 0.05

        # normalize بحيث المجموع = 1
        total = sum(base.values())
        return {k: round(v / total, 4) for k, v in base.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  RELIABILITY SCORER  — يحسب موثوقية السائق
# ══════════════════════════════════════════════════════════════════════════════

class ReliabilityScorer:

    @staticmethod
    def get_score(driver: dict) -> float:
        """
        0.0 → 1.0  
        يعتمد على:
          • acceptance_rate (نسبة قبول الطلبات)
          • cancellation_rate (نسبة الإلغاءات)
          • blacklist_until (محظور مؤقتاً؟)
        """
        # لو محظور مؤقتاً → صفر
        blacklist_until = driver.get("blacklist_until", 0)
        if blacklist_until and time.time() < blacklist_until:
            return 0.0

        accepted   = int(driver.get("total_accepted", 0))
        rejected   = int(driver.get("total_rejected", 0))
        cancelled  = int(driver.get("total_cancelled", 0))
        total_seen = accepted + rejected

        if total_seen == 0:
            acceptance_rate = 0.85  # افتراضي للسائق الجديد
        else:
            acceptance_rate = accepted / total_seen

        if accepted == 0:
            cancellation_rate = 0.0
        else:
            cancellation_rate = cancelled / accepted

        score = (acceptance_rate * 0.6) + ((1 - min(cancellation_rate, 1.0)) * 0.4)
        return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSIVENESS SCORER  — سرعة استجابة السائق للطلبات السابقة
# ══════════════════════════════════════════════════════════════════════════════

class ResponsivenessScorer:

    @staticmethod
    def get_score(driver: dict) -> float:
        """
        يعتمد على avg_response_seconds — متوسط وقت الرد على الطلبات.
        أقل من 30 ثانية → ممتاز
        30-90 ثانية → متوسط
        أكثر من 90 ثانية → ضعيف
        """
        avg_resp = float(driver.get("avg_response_seconds", 45.0))
        if avg_resp <= 30:
            return 1.0
        elif avg_resp <= 60:
            return 0.85
        elif avg_resp <= 90:
            return 0.65
        elif avg_resp <= 150:
            return 0.40
        else:
            return 0.20


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY AGENT  — المخ الرئيسي
# ══════════════════════════════════════════════════════════════════════════════

class UtilityAgent:
    """
    Utility-Based Agent:
    
    لكل سائق متاح، يحسب:
        U(driver) = Σ  wᵢ · fᵢ(driver, context)
    
    حيث:
        f1 = proximity_score   → (1 / (dist + 0.1)) normalized
        f2 = quality_score     → driver.rating / 5.0
        f3 = availability_score → 1 - (trips_today / MAX_TRIPS)
        f4 = reliability_score → acceptance_rate + cancellation_penalty
        f5 = speed_score       → 1 / avg_response_time normalized
    
    ثم يختار السائق ذو أعلى utility.
    """

    MAX_TRIPS_PER_DAY = 40
    MAX_PROXIMITY_DIST = 20.0   # كم — أبعد من كده مش مقبول

    @classmethod
    def _proximity_score(cls, dist_km: float) -> float:
        """
        Exponential decay — قريب جداً = 1.0، بعيد جداً ≈ 0
        """
        return round(math.exp(-0.3 * dist_km), 4)

    @classmethod
    def _quality_score(cls, driver: dict) -> float:
        rating = float(driver.get("rating", 4.5))
        # normalize: 1-star = 0, 5-star = 1, لكن نعاقب أقل من 3.5
        normalized = (rating - 1.0) / 4.0
        if rating < 3.5:
            normalized *= 0.5   # عقوبة على التقييم السيء
        return round(min(normalized, 1.0), 4)

    @classmethod
    def _availability_score(cls, driver: dict) -> float:
        trips_today = int(driver.get("trips_count_today", 0))
        load = min(trips_today, cls.MAX_TRIPS_PER_DAY) / cls.MAX_TRIPS_PER_DAY
        # سائق بـ 0 رحلات = 1.0، بـ 40 رحلة = 0.0
        return round(1.0 - load, 4)

    @classmethod
    def compute_utility(
        cls,
        driver: dict,
        user_lat: float,
        user_lon: float,
        weights: dict,
    ) -> dict[str, Any]:
        """يحسب الـ utility الكاملة ويرجع breakdown."""
        d_lat = float(driver.get("lat", user_lat))
        d_lon = float(driver.get("lon", user_lon))
        dist  = _haversine(user_lat, user_lon, d_lat, d_lon)

        if dist > cls.MAX_PROXIMITY_DIST:
            return {"total": -1.0, "dist_km": dist, "excluded": "too_far"}

        f1 = cls._proximity_score(dist)
        f2 = cls._quality_score(driver)
        f3 = cls._availability_score(driver)
        f4 = ReliabilityScorer.get_score(driver)
        f5 = ResponsivenessScorer.get_score(driver)

        # لو موثوقيته صفر (محظور) → استبعاد كامل
        if f4 == 0.0:
            return {"total": -1.0, "dist_km": dist, "excluded": "blacklisted"}

        total = (
            weights["w_proximity"]    * f1 +
            weights["w_quality"]      * f2 +
            weights["w_availability"] * f3 +
            weights["w_reliability"]  * f4 +
            weights["w_speed"]        * f5
        )

        return {
            "total": round(total, 6),
            "dist_km": round(dist, 2),
            "scores": {
                "proximity":    round(f1, 4),
                "quality":      round(f2, 4),
                "availability": round(f3, 4),
                "reliability":  round(f4, 4),
                "speed":        round(f5, 4),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SURGE PRICING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SurgePricingEngine:
    BASE_FARE    = 15.0    # جنيه
    PER_KM_FARE  = 8.0     # جنيه/كم

    @classmethod
    def calculate_price(cls, dist_km: float, demand_factor: float) -> dict:
        """
        السعر الديناميكي بناءً على المسافة والطلب.
        """
        base_price = cls.BASE_FARE + dist_km * cls.PER_KM_FARE

        # surge multiplier
        if demand_factor <= 0.5:
            multiplier = 0.9     # عرض عالي → خصم
        elif demand_factor <= 1.0:
            multiplier = 1.0     # عادي
        elif demand_factor <= 1.5:
            multiplier = 1.2     # طلب متوسط
        elif demand_factor <= 2.0:
            multiplier = 1.5     # ضغط
        elif demand_factor <= 3.0:
            multiplier = 1.8     # ضغط عالي
        else:
            multiplier = 2.2     # ضغط جداً

        # ليل = + 10%
        if ContextAnalyser.is_late_night():
            multiplier += 0.1

        final_price = round(base_price * multiplier, 0)

        return {
            "base_price":       round(base_price, 0),
            "multiplier":       round(multiplier, 2),
            "final_price":      final_price,
            "demand_factor":    demand_factor,
            "is_surge":         multiplier > 1.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  BLACKLIST MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class BlacklistManager:
    """
    لو السائق رفض 3 طلبات متتالية → محظور 15 دقيقة
    لو رفض 5 طلبات في يوم → محظور ساعة
    """

    @staticmethod
    def record_rejection(driver_id: str):
        ref = db.collection("users").document(driver_id)
        doc = ref.get()
        if not doc.exists:
            return

        data = doc.to_dict()
        consecutive_rejects = int(data.get("consecutive_rejects", 0)) + 1
        total_rejected = int(data.get("total_rejected", 0)) + 1

        updates: dict = {
            "consecutive_rejects": consecutive_rejects,
            "total_rejected":      total_rejected,
        }

        if consecutive_rejects >= 3:
            # محظور 15 دقيقة
            updates["blacklist_until"] = time.time() + 15 * 60
            updates["consecutive_rejects"] = 0
            print(f"⚠️  Driver {driver_id} blacklisted 15 min (3 consecutive rejects)")

        ref.update(updates)

    @staticmethod
    def record_acceptance(driver_id: str):
        db.collection("users").document(driver_id).update({
            "consecutive_rejects": 0,
            "total_accepted":      firestore.Increment(1),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def health():
    return {
        "status":  "ok",
        "service": "Harafy Utility-Based Agent",
        "version": "2.0.0",
        "time_cairo": ContextAnalyser.now_cairo().isoformat(),
        "is_rush_hour": ContextAnalyser.is_rush_hour(),
        "is_late_night": ContextAnalyser.is_late_night(),
    }


# ── match-driver ─────────────────────────────────────────────────────────────

@app.post("/match-driver", tags=["agent"])
async def match_driver(req: MatchRequest):
    """
    الـ AI Agent الرئيسي:
    1. يحلل السياق (وقت، طلب، انتظار)
    2. يحسب dynamic weights
    3. يحسب utility لكل سائق
    4. يختار الأعلى utility
    5. يرجع بيانات السائق + السعر الديناميكي
    """

    # 1. جلب السائقين المتاحين
    try:
        docs = (
            db.collection("users")
            .where("role", "==", "driver")
            .where("is_available", "==", True)
            .stream()
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Firestore error: {e}")

    # 2. تحليل السياق
    demand_factor = ContextAnalyser.get_demand_factor(req.user_lat, req.user_lon)

    # وقت الانتظار لو الرحلة موجودة
    wait_seconds = 0.0
    if req.trip_id:
        try:
            trip_doc = db.collection("trips").document(req.trip_id).get()
            if trip_doc.exists:
                created_at = trip_doc.to_dict().get("created_at")
                if created_at:
                    # Firestore timestamp → seconds
                    wait_seconds = time.time() - created_at.timestamp()
        except Exception:
            pass

    weights = ContextAnalyser.compute_dynamic_weights(demand_factor, wait_seconds)

    # 3. حساب utility لكل سائق
    candidates: list[dict] = []
    for doc in docs:
        data = doc.to_dict()
        data["_id"] = doc.id
        util = UtilityAgent.compute_utility(data, req.user_lat, req.user_lon, weights)
        if util["total"] < 0:
            continue   # مستبعد
        data["_utility"] = util
        candidates.append(data)

    if not candidates:
        raise HTTPException(status_code=404, detail="no_drivers_available")

    # 4. ترتيب حسب الـ utility وأخذ الأعلى
    candidates.sort(key=lambda x: x["_utility"]["total"], reverse=True)
    best = candidates[0]
    util_result = best["_utility"]

    # 5. حساب السعر الديناميكي
    dist_km     = util_result["dist_km"]
    pricing     = SurgePricingEngine.calculate_price(dist_km, demand_factor)
    eta         = _eta_minutes(dist_km)

    # 6. لو في trip_id → حدّث Firestore
    if req.trip_id:
        try:
            db.collection("trips").document(req.trip_id).update({
                "driver_id":          best["_id"],
                "driver_name":        best.get("name", "سائق"),
                "driver_phone":       best.get("phone", ""),
                "driver_lat":         best.get("lat"),
                "driver_lon":         best.get("lon"),
                "driver_rating":      best.get("rating", 5.0),
                "driver_car":         f"{best.get('car_brand','')} {best.get('car_model','')}".strip() or best.get("car", "سيارة"),
                "driver_car_color":   best.get("car_color", ""),
                "driver_plate":       best.get("plate_number", best.get("plate", "")),
                "driver_photo_url":   best.get("profile_photo_url", ""),
                "distance_km":        dist_km,
                "eta_min":            eta,
                "estimated_price":    pricing["final_price"],
                "surge_multiplier":   pricing["multiplier"],
                "is_surge":           pricing["is_surge"],
                "status":             "matched",
                "matched_at":         firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            print(f"⚠️ Firestore update error: {e}")

    return {
        "matched_driver": {
            "driver_id":      best["_id"],
            "name":           best.get("name", "سائق"),
            "phone":          best.get("phone", ""),
            "rating":         best.get("rating", 5.0),
            "car":            f"{best.get('car_brand','')} {best.get('car_model','')}".strip() or best.get("car", "سيارة"),
            "car_color":      best.get("car_color", ""),
            "plate":          best.get("plate_number", best.get("plate", "")),
            "photo_url":      best.get("profile_photo_url", ""),
            "lat":            best.get("lat"),
            "lon":            best.get("lon"),
            "distance_km":    dist_km,
            "eta_min":        eta,
        },
        "pricing":          pricing,
        "agent_decision": {
            "utility_score":  util_result["total"],
            "score_breakdown": util_result["scores"],
            "weights_used":   weights,
            "demand_factor":  demand_factor,
            "is_rush_hour":   ContextAnalyser.is_rush_hour(),
            "candidates_count": len(candidates),
        },
    }


# ── trip-status ───────────────────────────────────────────────────────────────

@app.post("/trip-status", tags=["trips"])
async def update_trip_status(req: TripStatusRequest):
    """
    يحدّث حالة الرحلة ويتعامل مع الـ side effects:
      accepted    → driver غير متاح
      driver_on_way → trigger ETA recalc
      arrived     → notify user
      completed   → حدّث earnings + trips_count + rating window
      cancelled   → driver متاح + blacklist check
    """
    valid_statuses = {"accepted", "driver_on_way", "arrived", "in_progress", "completed", "cancelled"}
    if req.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"invalid_status. Valid: {valid_statuses}")

    ref     = db.collection("trips").document(req.trip_id)
    trip    = ref.get()
    if not trip.exists:
        raise HTTPException(status_code=404, detail="trip_not_found")

    trip_data   = trip.to_dict()
    trip_driver = trip_data.get("driver_id")

    if trip_driver != req.driver_id:
        raise HTTPException(status_code=403, detail="not_your_trip")

    driver_ref = db.collection("users").document(req.driver_id)
    updates: dict = {
        "status":     req.status,
        f"{req.status}_at": firestore.SERVER_TIMESTAMP,
    }

    if req.status == "accepted":
        driver_ref.update({
            "is_available":       False,
            "consecutive_rejects": 0,
            "total_accepted":     firestore.Increment(1),
        })

    elif req.status == "driver_on_way":
        # لا يوجد side effect خاص — فقط timestamp
        pass

    elif req.status == "arrived":
        # إشعار للمستخدم يتم عبر Firestore listener في Flutter
        pass

    elif req.status == "in_progress":
        updates["trip_started_at"] = firestore.SERVER_TIMESTAMP

    elif req.status == "completed":
        price = float(trip_data.get("estimated_price", 0))
        driver_ref.update({
            "is_available":       True,
            "trips_count":        firestore.Increment(1),
            "trips_count_today":  firestore.Increment(1),
            "total_earnings":     firestore.Increment(price),
            "total_trips_ever":   firestore.Increment(1),
        })

    elif req.status == "cancelled":
        cancelled_by = trip_data.get("cancelled_by", "driver")
        driver_ref.update({"is_available": True})
        if cancelled_by == "driver":
            BlacklistManager.record_rejection(req.driver_id)

    ref.update(updates)

    return {
        "ok":      True,
        "trip_id": req.trip_id,
        "status":  req.status,
    }


# ── driver profile ────────────────────────────────────────────────────────────

@app.post("/driver/profile", tags=["profiles"])
async def save_driver_profile(req: DriverProfileRequest):
    """
    السائق يحدّث/يكمّل معلوماته الكاملة.
    هذه المعلومات تظهر للمستخدم أثناء الرحلة.
    """
    ref = db.collection("users").document(req.driver_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    ref.update({
        "name":             req.name,
        "phone":            req.phone,
        "car_brand":        req.car_brand,
        "car_model":        req.car_model,
        "car_year":         req.car_year,
        "car_color":        req.car_color,
        "plate_number":     req.plate_number,
        "national_id":      req.national_id,
        "profile_photo_url": req.profile_photo_url or "",
        "car_photo_url":    req.car_photo_url or "",
        # للعرض القديم
        "car":   f"{req.car_brand} {req.car_model}",
        "plate": req.plate_number,
        "profile_complete": True,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    return {"ok": True, "message": "تم حفظ بيانات السائق"}


# ── user profile ──────────────────────────────────────────────────────────────

@app.post("/user/profile", tags=["profiles"])
async def save_user_profile(req: UserProfileRequest):
    """المستخدم يكمّل معلوماته."""
    ref = db.collection("users").document(req.user_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="user_not_found")

    ref.update({
        "name":             req.name,
        "phone":            req.phone,
        "profile_photo_url": req.profile_photo_url or "",
        "profile_complete": True,
        "updated_at":       firestore.SERVER_TIMESTAMP,
    })

    return {"ok": True, "message": "تم حفظ بيانات المستخدم"}


# ── rating ────────────────────────────────────────────────────────────────────

@app.post("/trip/rate", tags=["trips"])
async def rate_driver(req: RatingRequest):
    """
    المستخدم يقيّم السائق بعد الرحلة.
    يحدث avg_rating بدل ما يحطها كـ static field.
    """
    driver_ref = db.collection("users").document(req.driver_id)
    doc = driver_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    data            = doc.to_dict()
    current_rating  = float(data.get("rating", 5.0))
    total_ratings   = int(data.get("total_ratings", 1))

    # Moving average
    new_rating = round(
        (current_rating * total_ratings + req.rating) / (total_ratings + 1), 2
    )

    driver_ref.update({
        "rating":        new_rating,
        "total_ratings": total_ratings + 1,
    })

    # احفظ التقييم في الرحلة
    db.collection("trips").document(req.trip_id).update({
        "user_rating":   req.rating,
        "user_comment":  req.comment or "",
        "rated_at":      firestore.SERVER_TIMESTAMP,
    })

    return {
        "ok":         True,
        "new_rating": new_rating,
        "total_ratings": total_ratings + 1,
    }


# ── driver stats ──────────────────────────────────────────────────────────────

@app.get("/driver/{driver_id}/stats", tags=["analytics"])
async def driver_stats(driver_id: str):
    doc = db.collection("users").document(driver_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")

    data = doc.to_dict()

    # حساب الـ reliability score الحالي
    reliability = ReliabilityScorer.get_score(data)
    responsiveness = ResponsivenessScorer.get_score(data)

    # هل محظور؟
    blacklist_until = data.get("blacklist_until", 0)
    blacklisted_for = max(0, round(blacklist_until - time.time()))

    return {
        "driver_id":           driver_id,
        "name":                data.get("name", "سائق"),
        "rating":              data.get("rating", 5.0),
        "trips_count":         data.get("trips_count", 0),
        "trips_count_today":   data.get("trips_count_today", 0),
        "total_earnings":      data.get("total_earnings", 0.0),
        "is_available":        data.get("is_available", False),
        "profile_complete":    data.get("profile_complete", False),
        "agent_scores": {
            "reliability":     reliability,
            "responsiveness":  responsiveness,
        },
        "blacklisted_seconds_remaining": blacklisted_for,
        "total_accepted":      data.get("total_accepted", 0),
        "total_rejected":      data.get("total_rejected", 0),
        "total_cancelled":     data.get("total_cancelled", 0),
    }


# ── surge pricing info ────────────────────────────────────────────────────────

@app.post("/surge-info", tags=["pricing"])
async def get_surge_info(req: SurgePricingRequest):
    """يحسب معامل الـ surge في منطقة معينة."""
    demand_factor = ContextAnalyser.get_demand_factor(req.lat, req.lon, req.radius_km)
    pricing_5km   = SurgePricingEngine.calculate_price(5.0, demand_factor)
    pricing_10km  = SurgePricingEngine.calculate_price(10.0, demand_factor)

    return {
        "demand_factor":   demand_factor,
        "is_surge":        pricing_5km["is_surge"],
        "surge_multiplier": pricing_5km["multiplier"],
        "is_rush_hour":    ContextAnalyser.is_rush_hour(),
        "is_late_night":   ContextAnalyser.is_late_night(),
        "sample_prices": {
            "5km":  pricing_5km["final_price"],
            "10km": pricing_10km["final_price"],
        },
    }


# ── area analytics ────────────────────────────────────────────────────────────

@app.get("/analytics/area", tags=["analytics"])
async def area_analytics(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(default=5.0, ge=0.5, le=20.0),
):
    """إحصائيات منطقة معينة — مفيد لـ dashboard."""
    demand_factor = ContextAnalyser.get_demand_factor(lat, lon, radius_km)

    # عدد السائقين المتاحين
    drivers_snap = (
        db.collection("users")
        .where("role", "==", "driver")
        .where("is_available", "==", True)
        .limit(100)
        .stream()
    )
    available_drivers = []
    for d in drivers_snap:
        data = d.to_dict()
        dlat = data.get("lat", 0)
        dlon = data.get("lon", 0)
        if _haversine(lat, lon, dlat, dlon) <= radius_km:
            available_drivers.append({
                "driver_id": d.id,
                "name":      data.get("name", "سائق"),
                "rating":    data.get("rating", 5.0),
                "lat":       dlat,
                "lon":       dlon,
            })

    weights = ContextAnalyser.compute_dynamic_weights(demand_factor)

    return {
        "area_center":          {"lat": lat, "lon": lon},
        "radius_km":            radius_km,
        "available_drivers":    len(available_drivers),
        "drivers_list":         available_drivers[:10],   # أول 10
        "demand_factor":        demand_factor,
        "current_weights":      weights,
        "is_rush_hour":         ContextAnalyser.is_rush_hour(),
        "is_late_night":        ContextAnalyser.is_late_night(),
        "surge_multiplier":     SurgePricingEngine.calculate_price(5.0, demand_factor)["multiplier"],
    }


# ── cancel trip (user side) ───────────────────────────────────────────────────

@app.post("/trip/{trip_id}/cancel", tags=["trips"])
async def cancel_trip(trip_id: str, cancelled_by: str = Query(default="user")):
    """إلغاء الرحلة — من المستخدم أو السائق."""
    ref  = db.collection("trips").document(trip_id)
    trip = ref.get()
    if not trip.exists:
        raise HTTPException(status_code=404, detail="trip_not_found")

    data = trip.to_dict()
    current_status = data.get("status")

    if current_status in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail="trip_already_finished")

    ref.update({
        "status":       "cancelled",
        "cancelled_by": cancelled_by,
        "cancelled_at": firestore.SERVER_TIMESTAMP,
    })

    # لو كان في سائق مخصص → حرّره
    driver_id = data.get("driver_id")
    if driver_id:
        db.collection("users").document(driver_id).update({"is_available": True})
        if cancelled_by == "driver":
            BlacklistManager.record_rejection(driver_id)

    return {"ok": True, "trip_id": trip_id, "cancelled_by": cancelled_by}


# ── driver location update ────────────────────────────────────────────────────

@app.put("/driver/{driver_id}/location", tags=["tracking"])
async def update_driver_location(
    driver_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    trip_id: str | None = Query(default=None),
):
    """
    تحديث موقع السائق (يُستدعى من Flutter كـ fallback لو Firebase offline).
    عادةً Flutter يحدّث Firestore مباشرة — هذا endpoint للـ edge cases.
    """
    updates = {
        "lat": lat,
        "lon": lon,
        "last_location_at": firestore.SERVER_TIMESTAMP,
    }
    db.collection("users").document(driver_id).update(updates)

    if trip_id:
        try:
            db.collection("trips").document(trip_id).update({
                "driver_lat": lat,
                "driver_lon": lon,
            })
        except Exception:
            pass

    return {"ok": True}

 
