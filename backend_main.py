"""
Harafy — AI Agent Backend  (نسخة مصلحة)
يشتغل على بورت 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import firebase_admin
from firebase_admin import credentials, firestore
import os

# ── تشغيل Firebase Admin ──────────────────────────────────────────────────
import json, tempfile

_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if _creds_json:
    # لو شغال على سيرفر (Render / Railway / VPS)
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    _tmp.write(_creds_json)
    _tmp.close()
    _SERVICE_KEY = _tmp.name
else:
    # لو شغال محلي
    _SERVICE_KEY = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(_SERVICE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(title="Harafy Agent", version="1.1.0")

# ✅ FIX 1: ضيّق الـ CORS للـ production
# غيّر "*" لـ domain بتاعك لما ترفع على سيرفر حقيقي
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: غيّره لـ ["https://yourdomain.com"] في production
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# ── Models ────────────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    user_id: str
    user_lat: float
    user_lon: float
    destination: str


class TripStatusRequest(BaseModel):
    trip_id: str
    status: str          # accepted | arrived | started | completed | cancelled
    driver_id: str


# ── Helpers ───────────────────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """المسافة بالكيلومتر بين نقطتين."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _utility_score(driver: dict, user_lat: float, user_lon: float) -> float:
    """
    Utility function للـ AI Agent:
        score = 0.5 * (1 / (dist+0.1))  +  0.3 * (rating/5)  +  0.2 * (1 - trips_today/50)
    """
    d_lat = driver.get("lat", user_lat)
    d_lon = driver.get("lon", user_lon)
    dist  = _haversine(user_lat, user_lon, d_lat, d_lon)

    rating     = float(driver.get("rating", 4.5))
    trips_done = int(driver.get("trips_count", 0))

    proximity = 1 / (dist + 0.1)
    score = 0.5 * proximity + 0.3 * (rating / 5.0) + 0.2 * (1 - min(trips_done, 50) / 50)
    return round(score, 4)


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "Harafy Agent", "version": "1.1.0"}


@app.post("/match-driver")
async def match_driver(req: MatchRequest):
    """
    يجيب كل السائقين المتاحين من Firestore
    ويطبّق عليهم Utility Function
    ويرجع أحسن سائق.
    """
    try:
        docs = (
            db.collection("users")
            .where("role", "==", "driver")
            .where("is_available", "==", True)
            .stream()
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Firestore error: {e}")

    drivers = []
    for doc in docs:
        data = doc.to_dict()
        data["driver_id"] = doc.id
        data["score"] = _utility_score(data, req.user_lat, req.user_lon)
        drivers.append(data)

    if not drivers:
        raise HTTPException(status_code=404, detail="no_drivers_available")

    best = sorted(drivers, key=lambda x: x["score"], reverse=True)[0]

    dist_km = _haversine(
        req.user_lat, req.user_lon,
        best.get("lat", req.user_lat),
        best.get("lon", req.user_lon),
    )

    # ✅ FIX 2: حساب الـ ETA صح
    # السرعة المتوسطة في الزحمة ≈ 20 كم/ساعة = 0.333 كم/دقيقة
    SPEED_KM_PER_MIN = 20 / 60
    eta_min = max(1, round(dist_km / SPEED_KM_PER_MIN))

    return {
        "matched_driver": {
            "driver_id":     best["driver_id"],
            "name":          best.get("name", "سائق"),
            "rating":        best.get("rating", 5.0),
            "distance_km":   round(dist_km, 1),
            "eta_min":       eta_min,
            "car":           best.get("car", "Toyota Camry"),
            "plate":         best.get("plate", "—"),
            "utility_score": best["score"],
        }
    }


@app.post("/trip-status")
async def update_trip_status(req: TripStatusRequest):
    """السائق يغيّر حالة الرحلة."""
    valid = {"accepted", "arrived", "started", "completed", "cancelled"}
    if req.status not in valid:
        raise HTTPException(status_code=400, detail="invalid_status")

    ref = db.collection("trips").document(req.trip_id)
    trip = ref.get()
    if not trip.exists:
        raise HTTPException(status_code=404, detail="trip_not_found")

    trip_data = trip.to_dict()

    # ✅ FIX 3: تأكد إن السائق ده هو فعلاً سائق الرحلة (أمان)
    if trip_data.get("driver_id") != req.driver_id:
        raise HTTPException(status_code=403, detail="not_your_trip")

    updates: dict = {"status": req.status}

    driver_ref = db.collection("users").document(req.driver_id)

    # ✅ FIX 4: إدارة is_available بشكل صح حسب حالة الرحلة
    if req.status == "accepted":
        # السائق اتقبل الرحلة — مش متاح لرحلات تانية
        driver_ref.update({"is_available": False})

    elif req.status == "completed":
        # الرحلة خلصت — زوّد العداد وخلّيه متاح تاني
        driver_ref.update({
            "trips_count": firestore.Increment(1),
            "is_available": True,
        })

    elif req.status == "cancelled":
        # الرحلة اتلغت — السائق يبقى متاح تاني
        driver_ref.update({"is_available": True})

    ref.update(updates)
    return {"ok": True, "trip_id": req.trip_id, "status": req.status}


@app.get("/driver/{driver_id}/stats")
async def driver_stats(driver_id: str):
    """إحصائيات السائق."""
    doc = db.collection("users").document(driver_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="driver_not_found")
    data = doc.to_dict()
    return {
        "trips_count":  data.get("trips_count", 0),
        "rating":       data.get("rating", 5.0),
        "is_available": data.get("is_available", False),
    }
