"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Geo & Pricing Module  —  Maps · Routing · Geocoding · Price        ║
║                          Harafy Backend  v2.1                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

كيفية الدمج مع main.py:
  1. ضع هذا الملف بجوار main.py
  2. في main.py أضف:
         from geo_pricing_module import (
             RouteEngine, GeocodingEngine, TripPricingEngine,
             PriceEstimateRequest, GeocodeRequest, ReverseGeocodeRequest,
             RouteRequest,
         )
  3. سجّل الـ routes بإضافة آخر الملف:
         app.include_router(geo_router)

المتطلبات (أضفها لـ requirements.txt):
  httpx>=0.27.0
  cachetools>=5.3.0
"""

from __future__ import annotations

import math
import time
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
#  Logger
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("geo_pricing")


# ══════════════════════════════════════════════════════════════════════════════
#  PRICING CONFIG  — عدّل هنا بدون لمس باقي الكود
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PricingConfig:
    """
    جميع متغيرات التسعير في مكان واحد.
    frozen=True يضمن عدم تعديلها عن طريق الخطأ أثناء التشغيل.
    """
    # ─── رسوم الأساس ───────────────────────────────────────────────
    base_fare_egp: float = 15.0         # رسوم ثابتة لكل رحلة
    per_km_fare_egp: float = 8.0        # جنيه لكل كيلومتر
    per_minute_fare_egp: float = 0.75   # جنيه لكل دقيقة في الرحلة
    minimum_fare_egp: float = 25.0      # أقل سعر مقبول

    # ─── البنزين ────────────────────────────────────────────────────
    fuel_price_per_liter_egp: float = 14.0   # سعر اللتر الحالي
    fuel_consumption_per_100km: float = 9.5  # استهلاك السيارة (لتر/100كم)
    fuel_cost_coverage_ratio: float = 0.65   # نسبة تغطية تكلفة البنزين

    # ─── Surge ──────────────────────────────────────────────────────
    surge_thresholds: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)
    surge_multipliers: tuple = (0.90, 1.00, 1.20, 1.50, 1.80, 2.20)
    late_night_bonus: float = 0.10      # +10% ليل
    rush_hour_bonus: float  = 0.05      # +5% ذروة

    # ─── خدمة ───────────────────────────────────────────────────────
    service_fee_egp: float = 3.0        # رسوم الخدمة الثابتة
    platform_commission: float = 0.20   # 20% عمولة المنصة


# Instance واحد يستخدمه كل الكود
PRICING = PricingConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class RouteRequest(BaseModel):
    origin_lat:  float = Field(..., ge=-90,  le=90)
    origin_lon:  float = Field(..., ge=-180, le=180)
    dest_lat:    float = Field(..., ge=-90,  le=90)
    dest_lon:    float = Field(..., ge=-180, le=180)
    demand_factor: float = Field(default=1.0, ge=0.1, le=5.0)


class PriceEstimateRequest(BaseModel):
    origin_lat:   float = Field(..., ge=-90,  le=90)
    origin_lon:   float = Field(..., ge=-180, le=180)
    dest_lat:     float = Field(..., ge=-90,  le=90)
    dest_lon:     float = Field(..., ge=-180, le=180)
    demand_factor: float = Field(default=1.0, ge=0.1, le=5.0)
    trip_id:      str | None = None   # لو موجود → يحدّث Firestore


class GeocodeRequest(BaseModel):
    address: str = Field(..., min_length=3, max_length=300)

    @field_validator("address")
    @classmethod
    def strip_address(cls, v: str) -> str:
        return v.strip()


class ReverseGeocodeRequest(BaseModel):
    lat: float = Field(..., ge=-90,  le=90)
    lon: float = Field(..., ge=-180, le=180)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE LAYER  — يقلل الطلبات المكررة على OSRM/Nominatim
# ══════════════════════════════════════════════════════════════════════════════

class _GeoCache:
    """
    Cache بسيط في الذاكرة بـ TTL.
    - Route cache: مدة 10 دقائق (الطرق ما بتتغيرش كثير)
    - Geocode cache: مدة 60 دقيقة (العناوين ثابتة)
    """
    _routes:   TTLCache = TTLCache(maxsize=500,  ttl=600)
    _geocodes: TTLCache = TTLCache(maxsize=1000, ttl=3600)

    @classmethod
    def route_key(cls, olat: float, olon: float, dlat: float, dlon: float) -> str:
        return f"{round(olat,4)},{round(olon,4)}-{round(dlat,4)},{round(dlon,4)}"

    @classmethod
    def geocode_key(cls, text: str) -> str:
        return text.lower().strip()

    @classmethod
    def get_route(cls, key: str) -> dict | None:
        return cls._routes.get(key)

    @classmethod
    def set_route(cls, key: str, value: dict) -> None:
        cls._routes[key] = value

    @classmethod
    def get_geocode(cls, key: str) -> dict | None:
        return cls._geocodes.get(key)

    @classmethod
    def set_geocode(cls, key: str, value: dict) -> None:
        cls._geocodes[key] = value


GeoCache = _GeoCache()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE ENGINE  — OSRM (مجاني تماماً، لا يحتاج API Key)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteResult:
    distance_km:   float
    duration_min:  float
    geometry:      list[list[float]]  # [[lon, lat], ...] للخريطة
    source:        str = "osrm"       # أو "haversine" كـ fallback


class RouteEngine:
    """
    يحسب المسار الفعلي بين نقطتين عبر OSRM.
    
    OSRM هو محرك توجيه مفتوح المصدر يستخدم بيانات OpenStreetMap.
    الـ public instance مناسب للتطوير والأحمال المتوسطة.
    للإنتاج يُنصح بنشر instance خاص.
    
    Fallback تلقائي: لو OSRM غير متاح → Haversine (خط مستقيم).
    """

    _OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"
    _TIMEOUT   = 5.0   # ثواني

    @classmethod
    async def get_route(
        cls,
        origin_lat: float, origin_lon: float,
        dest_lat: float,   dest_lon: float,
    ) -> RouteResult:
        """
        الدالة الرئيسية — تحسب المسار مع كاش تلقائي.
        """
        cache_key = GeoCache.route_key(origin_lat, origin_lon, dest_lat, dest_lon)
        cached = GeoCache.get_route(cache_key)
        if cached:
            return RouteResult(**cached)

        result = await cls._fetch_osrm(origin_lat, origin_lon, dest_lat, dest_lon)
        GeoCache.set_route(cache_key, {
            "distance_km":  result.distance_km,
            "duration_min": result.duration_min,
            "geometry":     result.geometry,
            "source":       result.source,
        })
        return result

    @classmethod
    async def _fetch_osrm(
        cls,
        olat: float, olon: float,
        dlat: float, dlon: float,
    ) -> RouteResult:
        url = f"{cls._OSRM_BASE}/{olon},{olat};{dlon},{dlat}"
        params = {
            "overview":    "simplified",   # geomety مبسطة للعرض
            "geometries":  "geojson",
            "steps":       "false",
        }
        try:
            async with httpx.AsyncClient(timeout=cls._TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                raise ValueError("OSRM: لم يُرجع مسار صالح")

            route = data["routes"][0]
            dist_km  = round(route["distance"] / 1000, 2)
            dur_min  = round(route["duration"] / 60, 1)
            coords   = route["geometry"]["coordinates"]   # [[lon,lat],...]

            return RouteResult(
                distance_km=dist_km,
                duration_min=dur_min,
                geometry=coords,
                source="osrm",
            )

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            log.warning(f"[RouteEngine] OSRM غير متاح، استخدام Haversine: {exc}")
            return cls._haversine_fallback(olat, olon, dlat, dlon)

    @staticmethod
    def _haversine_fallback(
        lat1: float, lon1: float, lat2: float, lon2: float,
    ) -> RouteResult:
        """
        حساب المسافة بخط مستقيم مع تعويض 1.35x للطرق الفعلية.
        """
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a    = (math.sin(dlat / 2) ** 2 +
                math.cos(math.radians(lat1)) *
                math.cos(math.radians(lat2)) *
                math.sin(dlon / 2) ** 2)
        straight = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        road_factor  = 1.35   # تعويض الانحناءات والطرق
        dist_km      = round(straight * road_factor, 2)
        speed_kmh    = 25.0   # متوسط سرعة في المدينة
        duration_min = round((dist_km / speed_kmh) * 60, 1)

        return RouteResult(
            distance_km=dist_km,
            duration_min=duration_min,
            geometry=[[lon1, lat1], [lon2, lat2]],
            source="haversine_fallback",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GEOCODING ENGINE  — Nominatim (OpenStreetMap, مجاني)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeocodeResult:
    lat:         float
    lon:         float
    display_name: str
    place_type:  str
    confidence:  float   # 0.0 → 1.0 — مدى دقة النتيجة


class GeocodingEngine:
    """
    تحويل العناوين النصية ← → إحداثيات باستخدام Nominatim.
    
    سياسة الاستخدام المقبول لـ Nominatim:
      • User-Agent إلزامي (محدد أدناه)
      • طلب واحد/ثانية كحد أقصى للـ public instance
      • للإنتاج: نشر instance خاص أو استخدام API مدفوع
    """

    _BASE     = "https://nominatim.openstreetmap.org"
    _HEADERS  = {"User-Agent": "Harafy-App/2.1 (harafy@example.com)"}
    _TIMEOUT  = 6.0

    # لتطبيق مصر: يضيق نطاق البحث لمصر تلقائياً
    _EGYPT_VIEWBOX  = "24.70,21.97,36.90,31.67"   # lon_min,lat_min,lon_max,lat_max
    _BOUNDED        = 1                             # 1 = مقيّد بالنطاق

    @classmethod
    async def geocode(cls, address: str) -> GeocodeResult:
        """عنوان نصي → إحداثيات."""
        key = GeoCache.geocode_key(f"fwd:{address}")
        cached = GeoCache.get_geocode(key)
        if cached:
            return GeocodeResult(**cached)

        result = await cls._fetch_forward(address)
        GeoCache.set_geocode(key, {
            "lat":          result.lat,
            "lon":          result.lon,
            "display_name": result.display_name,
            "place_type":   result.place_type,
            "confidence":   result.confidence,
        })
        return result

    @classmethod
    async def reverse_geocode(cls, lat: float, lon: float) -> GeocodeResult:
        """إحداثيات → عنوان نصي."""
        key = GeoCache.geocode_key(f"rev:{round(lat,5)},{round(lon,5)}")
        cached = GeoCache.get_geocode(key)
        if cached:
            return GeocodeResult(**cached)

        result = await cls._fetch_reverse(lat, lon)
        GeoCache.set_geocode(key, {
            "lat":          result.lat,
            "lon":          result.lon,
            "display_name": result.display_name,
            "place_type":   result.place_type,
            "confidence":   result.confidence,
        })
        return result

    @classmethod
    async def _fetch_forward(cls, address: str) -> GeocodeResult:
        params = {
            "q":          address,
            "format":     "jsonv2",
            "limit":      1,
            "viewbox":    cls._EGYPT_VIEWBOX,
            "bounded":    cls._BOUNDED,
            "addressdetails": 1,
        }
        try:
            async with httpx.AsyncClient(
                headers=cls._HEADERS, timeout=cls._TIMEOUT
            ) as client:
                resp = await client.get(f"{cls._BASE}/search", params=params)
                resp.raise_for_status()
                results = resp.json()

            if not results:
                # حاول بدون حدود مصر
                return await cls._fetch_forward_global(address)

            r = results[0]
            return GeocodeResult(
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                display_name=r.get("display_name", ""),
                place_type=r.get("type", "unknown"),
                confidence=min(float(r.get("importance", 0.5)), 1.0),
            )

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"خدمة الخرائط غير متاحة مؤقتاً: {exc}",
            )

    @classmethod
    async def _fetch_forward_global(cls, address: str) -> GeocodeResult:
        """محاولة ثانية بدون تقييد جغرافي."""
        params = {"q": address, "format": "jsonv2", "limit": 1}
        try:
            async with httpx.AsyncClient(
                headers=cls._HEADERS, timeout=cls._TIMEOUT
            ) as client:
                resp = await client.get(f"{cls._BASE}/search", params=params)
                resp.raise_for_status()
                results = resp.json()

            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="لم يتم العثور على العنوان — حاول بصياغة أوضح",
                )
            r = results[0]
            return GeocodeResult(
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                display_name=r.get("display_name", ""),
                place_type=r.get("type", "unknown"),
                confidence=min(float(r.get("importance", 0.3)), 1.0),
            )
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @classmethod
    async def _fetch_reverse(cls, lat: float, lon: float) -> GeocodeResult:
        params = {
            "lat":    lat,
            "lon":    lon,
            "format": "jsonv2",
            "zoom":   16,   # دقة الحي
        }
        try:
            async with httpx.AsyncClient(
                headers=cls._HEADERS, timeout=cls._TIMEOUT
            ) as client:
                resp = await client.get(f"{cls._BASE}/reverse", params=params)
                resp.raise_for_status()
                r = resp.json()

            if "error" in r:
                raise HTTPException(
                    status_code=404,
                    detail="لا يوجد عنوان لهذه الإحداثيات",
                )
            return GeocodeResult(
                lat=lat,
                lon=lon,
                display_name=r.get("display_name", ""),
                place_type=r.get("type", "unknown"),
                confidence=0.95,   # reverse دائماً عالي الدقة
            )
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  TRIP PRICING ENGINE  — محرك التسعير الكامل
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PriceBreakdown:
    """تفصيل كامل لمكونات السعر."""
    base_fare:       float   # رسوم الانطلاق
    distance_cost:   float   # تكلفة المسافة
    time_cost:       float   # تكلفة الوقت
    fuel_component:  float   # حصة البنزين
    service_fee:     float   # رسوم الخدمة
    subtotal:        float   # المجموع قبل الـ surge
    surge_multiplier: float
    surge_reason:    str     # وصف سبب الـ surge
    final_price:     float   # السعر النهائي (مقرّب)
    driver_earnings: float   # نصيب السائق
    is_surge:        bool
    distance_km:     float
    duration_min:    float
    source:          str     # osrm | haversine_fallback


class TripPricingEngine:
    """
    محرك التسعير الديناميكي.
    
    المعادلة:
        subtotal = base_fare + (dist_km × per_km) + (duration_min × per_min)
                 + fuel_component + service_fee
        final    = subtotal × surge_multiplier
        final    = max(final, minimum_fare)
    
    fuel_component:
        يحاكي تكلفة البنزين الفعلية ويغطي جزءاً منها تلقائياً.
        fuel_cost = (dist_km / 100) × consumption × price_per_liter
        fuel_component = fuel_cost × coverage_ratio
    """

    @classmethod
    def calculate(
        cls,
        route: RouteResult,
        demand_factor: float,
        is_rush_hour: bool = False,
        is_late_night: bool = False,
    ) -> PriceBreakdown:

        cfg = PRICING

        # ─── المكونات الأساسية ────────────────────────────────────────
        base_fare      = cfg.base_fare_egp
        distance_cost  = round(route.distance_km * cfg.per_km_fare_egp, 2)
        time_cost      = round(route.duration_min * cfg.per_minute_fare_egp, 2)
        service_fee    = cfg.service_fee_egp

        # ─── مكوّن البنزين ────────────────────────────────────────────
        raw_fuel_cost  = (route.distance_km / 100) * cfg.fuel_consumption_per_100km * cfg.fuel_price_per_liter_egp
        fuel_component = round(raw_fuel_cost * cfg.fuel_cost_coverage_ratio, 2)

        subtotal = round(base_fare + distance_cost + time_cost + fuel_component + service_fee, 2)

        # ─── Surge Multiplier ─────────────────────────────────────────
        multiplier, surge_reason = cls._compute_surge(demand_factor, is_rush_hour, is_late_night)

        # ─── السعر النهائي ────────────────────────────────────────────
        raw_final  = subtotal * multiplier
        final_price = max(round(raw_final / 5) * 5, cfg.minimum_fare_egp)   # تقريب لأقرب 5 جنيه

        # ─── نصيب السائق ─────────────────────────────────────────────
        driver_earnings = round(final_price * (1 - cfg.platform_commission), 1)

        return PriceBreakdown(
            base_fare=base_fare,
            distance_cost=distance_cost,
            time_cost=time_cost,
            fuel_component=fuel_component,
            service_fee=service_fee,
            subtotal=subtotal,
            surge_multiplier=round(multiplier, 2),
            surge_reason=surge_reason,
            final_price=float(final_price),
            driver_earnings=driver_earnings,
            is_surge=multiplier > 1.0,
            distance_km=route.distance_km,
            duration_min=route.duration_min,
            source=route.source,
        )

    @staticmethod
    def _compute_surge(
        demand_factor: float,
        is_rush_hour: bool,
        is_late_night: bool,
    ) -> tuple[float, str]:
        """
        يحسب معامل الـ surge وسبب تطبيقه.
        """
        cfg = PRICING
        thresholds   = cfg.surge_thresholds
        multipliers  = cfg.surge_multipliers

        # تحديد المعامل الأساسي
        multiplier = multipliers[0]   # أقل قيمة (خصم)
        for i, threshold in enumerate(thresholds):
            if demand_factor > threshold:
                multiplier = multipliers[i + 1]

        reasons = []
        if multiplier > 1.0:
            reasons.append(f"ضغط الطلب ({demand_factor:.1f}x)")
        elif multiplier < 1.0:
            reasons.append("عرض مرتفع — خصم تلقائي")

        if is_late_night:
            multiplier = round(multiplier + cfg.late_night_bonus, 2)
            reasons.append("رحلة ليلية")

        if is_rush_hour:
            multiplier = round(multiplier + cfg.rush_hour_bonus, 2)
            reasons.append("ساعة الذروة")

        surge_reason = " + ".join(reasons) if reasons else "سعر عادي"
        return multiplier, surge_reason

    @classmethod
    def to_firestore_dict(cls, breakdown: PriceBreakdown) -> dict:
        """يحوّل الـ breakdown لصيغة Firestore-ready."""
        return {
            "estimated_price":      breakdown.final_price,
            "price_breakdown": {
                "base_fare":        breakdown.base_fare,
                "distance_cost":    breakdown.distance_cost,
                "time_cost":        breakdown.time_cost,
                "fuel_component":   breakdown.fuel_component,
                "service_fee":      breakdown.service_fee,
                "subtotal":         breakdown.subtotal,
            },
            "surge_multiplier":     breakdown.surge_multiplier,
            "surge_reason":         breakdown.surge_reason,
            "is_surge":             breakdown.is_surge,
            "driver_earnings":      breakdown.driver_earnings,
            "distance_km":          breakdown.distance_km,
            "duration_min":         breakdown.duration_min,
            "route_source":         breakdown.source,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER  — الـ Endpoints الجديدة
# ══════════════════════════════════════════════════════════════════════════════

geo_router = APIRouter(prefix="/geo", tags=["geo & pricing"])


# ── 1. حساب المسار الكامل ────────────────────────────────────────────────────

@geo_router.post("/route")
async def get_route(req: RouteRequest):
    """
    يحسب المسار الفعلي بين نقطتين + السعر الديناميكي.

    الاستخدام من Flutter:
        POST /geo/route
        Body: { origin_lat, origin_lon, dest_lat, dest_lon, demand_factor }
    
    الرد يحتوي:
        • distance_km: المسافة الحقيقية عبر الطرق
        • duration_min: الوقت المتوقع
        • geometry: إحداثيات المسار لرسمه على الخريطة
        • pricing: السعر الكامل مع التفاصيل
    """
    try:
        route = await RouteEngine.get_route(
            req.origin_lat, req.origin_lon,
            req.dest_lat,   req.dest_lon,
        )
    except Exception as exc:
        log.error(f"[/geo/route] {exc}")
        raise HTTPException(status_code=503, detail="خطأ في حساب المسار")

    # نستورد من main.py للتحقق من الوقت
    from datetime import timezone, timedelta, datetime
    cairo_tz = timezone(timedelta(hours=2))
    now_h = datetime.now(cairo_tz).hour
    is_rush  = any(s <= now_h < e for s, e in [(7, 10), (16, 20)])
    is_night = now_h >= 23 or now_h < 5

    pricing = TripPricingEngine.calculate(route, req.demand_factor, is_rush, is_night)

    return {
        "route": {
            "distance_km":  route.distance_km,
            "duration_min": route.duration_min,
            "geometry":     route.geometry,
            "source":       route.source,
        },
        "pricing": {
            "final_price":      pricing.final_price,
            "driver_earnings":  pricing.driver_earnings,
            "is_surge":         pricing.is_surge,
            "surge_multiplier": pricing.surge_multiplier,
            "surge_reason":     pricing.surge_reason,
            "breakdown": {
                "base_fare":      pricing.base_fare,
                "distance_cost":  pricing.distance_cost,
                "time_cost":      pricing.time_cost,
                "fuel_component": pricing.fuel_component,
                "service_fee":    pricing.service_fee,
                "subtotal":       pricing.subtotal,
            },
        },
    }


# ── 2. تقدير السعر فقط + حفظ في Firestore ───────────────────────────────────

@geo_router.post("/price-estimate")
async def price_estimate(req: PriceEstimateRequest):
    """
    يحسب السعر الديناميكي بالكامل ويحدّث Firestore لحظياً.

    لو أرسلت trip_id → يُكتب السعر في وثيقة الرحلة مباشرة.
    الطرفان (السائق والمستخدم) يشوفانه في نفس اللحظة عبر الـ listener.
    """
    try:
        route = await RouteEngine.get_route(
            req.origin_lat, req.origin_lon,
            req.dest_lat,   req.dest_lon,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"خطأ مسار: {exc}")

    from datetime import timezone, timedelta, datetime
    cairo_tz = timezone(timedelta(hours=2))
    now_h = datetime.now(cairo_tz).hour
    is_rush  = any(s <= now_h < e for s, e in [(7, 10), (16, 20)])
    is_night = now_h >= 23 or now_h < 5

    pricing = TripPricingEngine.calculate(route, req.demand_factor, is_rush, is_night)

    # ─── حفظ في Firestore لو في trip_id ──────────────────────────────
    if req.trip_id:
        try:
            from firebase_admin import firestore as fs
            db_ref = fs.client()
            updates = TripPricingEngine.to_firestore_dict(pricing)
            updates["price_updated_at"] = fs.SERVER_TIMESTAMP
            db_ref.collection("trips").document(req.trip_id).update(updates)
            log.info(f"[price_estimate] trip={req.trip_id} price={pricing.final_price}")
        except Exception as exc:
            log.warning(f"[price_estimate] Firestore update failed: {exc}")
            # لا نوقف الرد — نكمل حتى لو Firestore فشل

    return {
        "final_price":      pricing.final_price,
        "driver_earnings":  pricing.driver_earnings,
        "is_surge":         pricing.is_surge,
        "surge_multiplier": pricing.surge_multiplier,
        "surge_reason":     pricing.surge_reason,
        "distance_km":      pricing.distance_km,
        "duration_min":     pricing.duration_min,
        "breakdown": {
            "base_fare":      pricing.base_fare,
            "distance_cost":  pricing.distance_cost,
            "time_cost":      pricing.time_cost,
            "fuel_component": pricing.fuel_component,
            "service_fee":    pricing.service_fee,
            "subtotal":       pricing.subtotal,
        },
        "route_source": pricing.source,
        "firestore_updated": bool(req.trip_id),
    }


# ── 3. Geocoding: عنوان → إحداثيات ──────────────────────────────────────────

@geo_router.post("/geocode")
async def geocode_address(req: GeocodeRequest):
    """
    يحوّل العنوان النصي (مثل 'مدينة نصر، القاهرة') إلى إحداثيات.

    مفيد لـ:
      • تسجيل الوجهة بالكتابة بدل التحديد على الخريطة
      • حفظ مواقع المنزل / العمل
    """
    result = await GeocodingEngine.geocode(req.address)
    return {
        "lat":          result.lat,
        "lon":          result.lon,
        "display_name": result.display_name,
        "place_type":   result.place_type,
        "confidence":   result.confidence,
    }


# ── 4. Reverse Geocoding: إحداثيات → عنوان ──────────────────────────────────

@geo_router.post("/reverse-geocode")
async def reverse_geocode(req: ReverseGeocodeRequest):
    """
    يحوّل الإحداثيات (GPS) إلى عنوان نصي قابل للقراءة.

    مفيد لـ:
      • عرض موقع المستخدم / السائق كعنوان بدل أرقام
      • تسمية نقطة الانطلاق تلقائياً من GPS
    """
    result = await GeocodingEngine.reverse_geocode(req.lat, req.lon)
    return {
        "display_name": result.display_name,
        "place_type":   result.place_type,
        "lat":          result.lat,
        "lon":          result.lon,
    }


# ── 5. Batch Geocode: عناوين متعددة دفعة واحدة ──────────────────────────────

class BatchGeocodeRequest(BaseModel):
    addresses: list[str] = Field(..., min_length=1, max_length=20)


@geo_router.post("/geocode/batch")
async def batch_geocode(req: BatchGeocodeRequest):
    """
    يحوّل قائمة من العناوين إلى إحداثيات بطلب واحد.
    مفيد لتحميل المواقع المحفوظة (المنزل، العمل، إلخ) دفعة واحدة.
    
    ملاحظة: الطلبات تُنفَّذ بتوازٍ مع delay بسيط لاحترام rate limit Nominatim.
    """
    results = []
    for address in req.addresses:
        try:
            geo = await GeocodingEngine.geocode(address)
            results.append({
                "address":      address,
                "lat":          geo.lat,
                "lon":          geo.lon,
                "display_name": geo.display_name,
                "confidence":   geo.confidence,
                "status":       "ok",
            })
        except HTTPException as exc:
            results.append({
                "address": address,
                "status":  "error",
                "detail":  exc.detail,
            })
        await asyncio.sleep(0.25)   # احترام Nominatim rate limit

    return {"results": results, "count": len(results)}


# ── 6. Pricing Config (للعرض فقط — لو المشرف أراد التحقق) ──────────────────

@geo_router.get("/pricing-config")
def get_pricing_config():
    """
    يعرض إعدادات التسعير الحالية.
    مفيد للـ dashboard الإداري.
    """
    cfg = PRICING
    return {
        "base_fare_egp":              cfg.base_fare_egp,
        "per_km_fare_egp":            cfg.per_km_fare_egp,
        "per_minute_fare_egp":        cfg.per_minute_fare_egp,
        "minimum_fare_egp":           cfg.minimum_fare_egp,
        "service_fee_egp":            cfg.service_fee_egp,
        "platform_commission_pct":    cfg.platform_commission * 100,
        "fuel_price_per_liter_egp":   cfg.fuel_price_per_liter_egp,
        "fuel_consumption_l_100km":   cfg.fuel_consumption_per_100km,
        "fuel_coverage_ratio_pct":    cfg.fuel_cost_coverage_ratio * 100,
        "surge_late_night_bonus_pct": cfg.late_night_bonus * 100,
        "surge_rush_hour_bonus_pct":  cfg.rush_hour_bonus * 100,
        "note": "عدّل PricingConfig في geo_pricing_module.py لتغيير الأسعار",
    }
