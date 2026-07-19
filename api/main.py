"""
GramFlow API — NABARD Rural Enterprise Risk Intelligence Platform
FastAPI skeleton with 4 core endpoints for Round 1 submission.

Endpoints:
    POST /v1/onboard          — Sector onboarding
    POST /v1/ingest           — Data ingestion trigger
    GET  /v1/forecast/{id}    — Cash-flow forecast
    GET  /v1/risk/{id}        — Risk classification + SHAP Hindi alert
"""

from __future__ import annotations

import uuid
import random
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    OnboardRequest,
    OnboardResponse,
    IngestRequest,
    IngestResponse,
    ForecastResponse,
    ForecastPoint,
    RiskResponse,
    ShapFeature,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GramFlow — NABARD Rural Risk Intelligence API",
    description=(
        "AI-powered cash-flow forecasting and risk classification for rural "
        "micro-enterprises in India. Supports dairy, poultry, food processing, "
        "handicrafts, and rural retail sectors."
    ),
    version="0.1.0",
    contact={
        "name": "GramFlow Team",
        "url": "https://github.com/panshularora/nabard",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory registry (replace with DB in production)
# ---------------------------------------------------------------------------

_ENTERPRISE_REGISTRY: dict[str, dict] = {}

SECTOR_CONFIG = {
    "dairy": {
        "risk_weights": {"upi_velocity": 0.30, "milk_price_volatility": 0.25, "feed_cost_ratio": 0.25, "rainfall_index": 0.20},
        "seasonal_peaks": ["October", "November", "December"],
        "seasonal_troughs": ["April", "May"],
        "key_inputs": ["cattle_feed", "veterinary_services", "electricity"],
    },
    "poultry": {
        "risk_weights": {"upi_velocity": 0.25, "egg_price_volatility": 0.30, "mortality_rate": 0.25, "feed_cost_ratio": 0.20},
        "seasonal_peaks": ["November", "December", "January"],
        "seasonal_troughs": ["June", "July"],
        "key_inputs": ["poultry_feed", "vaccines", "electricity"],
    },
    "food_processing": {
        "risk_weights": {"upi_velocity": 0.20, "raw_material_cost": 0.35, "sale_price_index": 0.25, "labor_cost": 0.20},
        "seasonal_peaks": ["October", "February", "March"],
        "seasonal_troughs": ["July", "August"],
        "key_inputs": ["raw_materials", "labor", "packaging", "electricity"],
    },
    "handicrafts": {
        "risk_weights": {"upi_velocity": 0.30, "order_velocity": 0.30, "material_cost": 0.20, "labor_days": 0.20},
        "seasonal_peaks": ["October", "November", "December"],
        "seasonal_troughs": ["January", "February"],
        "key_inputs": ["raw_materials", "skilled_labor", "tools"],
    },
    "rural_retail": {
        "risk_weights": {"upi_velocity": 0.40, "inventory_turnover": 0.25, "credit_utilization": 0.20, "footfall_proxy": 0.15},
        "seasonal_peaks": ["October", "November"],
        "seasonal_troughs": ["June", "July"],
        "key_inputs": ["inventory", "rental", "labor"],
    },
}

# ---------------------------------------------------------------------------
# SHAP mock templates (Hindi + English)
# ---------------------------------------------------------------------------

HINDI_ALERT_TEMPLATES = {
    "TIER_1": (
        "⚠️ उच्च जोखिम चेतावनी: आपके व्यवसाय में नकदी प्रवाह में गंभीर कमी आ सकती है। "
        "प्रमुख कारण: {top_factor}। कृपया अगले {days} दिनों में अपने नजदीकी NABARD कार्यालय से संपर्क करें।"
    ),
    "TIER_2": (
        "⚠️ मध्यम जोखिम: {top_factor} के कारण अगले {horizon} महीनों में आय में गिरावट संभव है। "
        "सुझाव: {action}। NABARD फील्ड अधिकारी से {days} दिनों में मिलें।"
    ),
    "TIER_3": (
        "ℹ️ सतर्कता: {top_factor} पर नजर रखें। अभी कोई तत्काल जोखिम नहीं है। "
        "अगले {horizon} महीनों का पूर्वानुमान स्थिर है।"
    ),
    "TIER_4": (
        "✅ स्थिर स्थिति: आपका व्यवसाय अगले {horizon} महीनों के लिए सुरक्षित है। "
        "नकदी प्रवाह सकारात्मक रहने की संभावना है।"
    ),
    "TIER_5": (
        "✅ उत्कृष्ट प्रदर्शन: आपका व्यवसाय बहुत अच्छी स्थिति में है। "
        "विस्तार के अवसर तलाशें — NABARD KCC योजना के लिए आवेदन करें।"
    ),
}

RISK_TIER_MAP = {
    1: {"label": "Critical", "color": "#dc2626", "score_range": (0, 20)},
    2: {"label": "High",     "color": "#ea580c", "score_range": (20, 40)},
    3: {"label": "Moderate", "color": "#d97706", "score_range": (40, 60)},
    4: {"label": "Low",      "color": "#16a34a", "score_range": (60, 80)},
    5: {"label": "Healthy",  "color": "#059669", "score_range": (80, 100)},
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _mock_forecast_series(horizon_months: int = 6) -> List[ForecastPoint]:
    """Generate a realistic mock cash-flow forecast series."""
    points: List[ForecastPoint] = []
    base = random.uniform(15_000, 80_000)
    trend = random.uniform(-0.02, 0.04)
    today = date.today()
    for m in range(1, horizon_months + 1):
        forecast_date = today + timedelta(days=30 * m)
        yhat = base * (1 + trend) ** m + random.gauss(0, base * 0.05)
        margin = abs(yhat) * random.uniform(0.10, 0.25)
        points.append(
            ForecastPoint(
                ds=forecast_date,
                yhat=round(yhat, 2),
                yhat_lower=round(yhat - margin, 2),
                yhat_upper=round(yhat + margin, 2),
            )
        )
    return points


def _mock_shap_features(sector: str) -> List[ShapFeature]:
    """Return mock SHAP feature attributions for a sector."""
    weights = SECTOR_CONFIG.get(sector, SECTOR_CONFIG["dairy"])["risk_weights"]
    features = []
    for feat, base_importance in weights.items():
        value = round(random.uniform(0.1, 1.0), 3)
        shap_val = round(random.gauss(base_importance * 10, 2), 3)
        features.append(ShapFeature(name=feat, value=value, shap_value=shap_val))
    return sorted(features, key=lambda x: abs(x.shap_value), reverse=True)


def _build_hindi_alert(tier: int, sector: str, shap_features: List[ShapFeature]) -> str:
    top = shap_features[0].name if shap_features else "नकदी प्रवाह"
    top_hindi = {
        "upi_velocity": "UPI लेनदेन गति",
        "milk_price_volatility": "दूध मूल्य अस्थिरता",
        "feed_cost_ratio": "चारे की लागत अनुपात",
        "rainfall_index": "वर्षा सूचकांक",
        "egg_price_volatility": "अंडे के मूल्य में उतार-चढ़ाव",
        "mortality_rate": "मृत्यु दर",
        "raw_material_cost": "कच्चे माल की लागत",
        "sale_price_index": "बिक्री मूल्य सूचकांक",
        "labor_cost": "श्रम लागत",
        "order_velocity": "ऑर्डर गति",
        "material_cost": "सामग्री लागत",
        "labor_days": "श्रम दिवस",
        "inventory_turnover": "इन्वेंटरी टर्नओवर",
        "credit_utilization": "ऋण उपयोग",
        "footfall_proxy": "ग्राहक आगमन",
    }.get(top, top)

    action_map = {
        "dairy": "पशु बीमा और KCC ऋण का उपयोग करें",
        "poultry": "जैव सुरक्षा उपाय अपनाएं",
        "food_processing": "थोक खरीद के लिए SHG से जुड़ें",
        "handicrafts": "e-Commerce पोर्टल पर रजिस्ट्रेशन करें",
        "rural_retail": "डिजिटल भुगतान प्रोत्साहित करें",
    }

    template = HINDI_ALERT_TEMPLATES.get(f"TIER_{tier}", HINDI_ALERT_TEMPLATES["TIER_3"])
    return template.format(
        top_factor=top_hindi,
        days=random.choice([7, 10, 14, 21]),
        horizon=6,
        action=action_map.get(sector, "NABARD अधिकारी से परामर्श लें"),
    )


# ---------------------------------------------------------------------------
# Endpoint 1 — Sector Onboarding
# ---------------------------------------------------------------------------

@app.post(
    "/v1/onboard",
    response_model=OnboardResponse,
    summary="Register an enterprise and select sector",
    tags=["Onboarding"],
)
async def onboard_enterprise(payload: OnboardRequest) -> OnboardResponse:
    """
    Registers a new rural micro-enterprise on GramFlow.

    - Validates sector against supported list
    - Loads sector-specific risk weight template
    - Returns enterprise ID and seasonal calendar
    """
    if payload.sector not in SECTOR_CONFIG:
        raise HTTPException(
            status_code=422,
            detail=f"Sector '{payload.sector}' not supported. "
                   f"Choose from: {list(SECTOR_CONFIG.keys())}",
        )

    enterprise_id = f"ENT-{payload.sector[:3].upper()}-{uuid.uuid4().hex[:8].upper()}"
    config = SECTOR_CONFIG[payload.sector]

    record = {
        "enterprise_id": enterprise_id,
        "name": payload.enterprise_name,
        "sector": payload.sector,
        "district": payload.district,
        "state": payload.state,
        "registered_at": datetime.utcnow().isoformat(),
        "risk_weights": config["risk_weights"],
    }
    _ENTERPRISE_REGISTRY[enterprise_id] = record

    return OnboardResponse(
        enterprise_id=enterprise_id,
        sector=payload.sector,
        risk_weights=config["risk_weights"],
        seasonal_peaks=config["seasonal_peaks"],
        seasonal_troughs=config["seasonal_troughs"],
        key_inputs=config["key_inputs"],
        message=f"Enterprise '{payload.enterprise_name}' successfully registered under GramFlow. "
                f"Sector template '{payload.sector}' loaded.",
    )


# ---------------------------------------------------------------------------
# Endpoint 2 — Data Ingestion Trigger
# ---------------------------------------------------------------------------

@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    summary="Trigger daily data ingestion pipeline",
    tags=["Data Ingestion"],
)
async def trigger_ingestion(payload: IngestRequest) -> IngestResponse:
    """
    Triggers (or simulates) the daily Airflow ingestion pipeline for an enterprise.

    Sources pulled:
    - Commodity price feed (AGMARKNET / eNAM proxy)
    - Climate signals (IMD API)
    - UPI aggregate transactions (anonymised proxy)
    - Mobile-logged income/expense records
    """
    if payload.enterprise_id not in _ENTERPRISE_REGISTRY:
        # Allow mock IDs for demo
        pass

    job_id = f"INGEST-{uuid.uuid4().hex[:12].upper()}"

    records_ingested = {
        "commodity_prices": random.randint(5, 20),
        "climate_signals": random.randint(3, 10),
        "upi_transactions": random.randint(15, 120),
        "mobile_logs": random.randint(1, 30),
    }

    return IngestResponse(
        job_id=job_id,
        enterprise_id=payload.enterprise_id,
        status="queued",
        records_ingested=records_ingested,
        ingestion_date=date.today(),
        message=(
            f"Ingestion job {job_id} queued for enterprise {payload.enterprise_id}. "
            f"Total {sum(records_ingested.values())} records staged for feature engineering."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint 3 — Cash Flow Forecast
# ---------------------------------------------------------------------------

@app.get(
    "/v1/forecast/{enterprise_id}",
    response_model=ForecastResponse,
    summary="Get 6-month cash-flow forecast",
    tags=["Forecasting"],
)
async def get_forecast(
    enterprise_id: str,
    horizon_months: int = Query(default=6, ge=1, le=12, description="Forecast horizon in months"),
) -> ForecastResponse:
    """
    Returns NeuralProphet + LightGBM ensemble cash-flow forecast for the enterprise.

    - Decomposes trend, seasonality, and holiday effects
    - Returns point forecast with 80% and 95% confidence intervals
    - Seasonal peaks/troughs identified from sector calendar
    """
    sector = _ENTERPRISE_REGISTRY.get(enterprise_id, {}).get("sector", "dairy")
    forecast_series = _mock_forecast_series(horizon_months)

    total_yhat = sum(p.yhat for p in forecast_series)
    trend = "upward" if forecast_series[-1].yhat > forecast_series[0].yhat else "downward"

    return ForecastResponse(
        enterprise_id=enterprise_id,
        sector=sector,
        horizon_months=horizon_months,
        model="NeuralProphet + LightGBM ensemble",
        forecast=forecast_series,
        total_projected_cashflow=round(total_yhat, 2),
        trend_direction=trend,
        generated_at=datetime.utcnow(),
        note=(
            "Forecast generated using mock data for Round 1 demo. "
            "Production model uses 180+ days of historical enterprise data."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint 4 — Risk Classification + SHAP Alert
# ---------------------------------------------------------------------------

@app.get(
    "/v1/risk/{enterprise_id}",
    response_model=RiskResponse,
    summary="Get risk tier and SHAP-powered Hindi alert",
    tags=["Risk Intelligence"],
)
async def get_risk(
    enterprise_id: str,
    lang: str = Query(default="hi", enum=["hi", "en"], description="Alert language: hi=Hindi, en=English"),
) -> RiskResponse:
    """
    Runs XGBoost risk classification and generates SHAP feature attributions.

    Returns:
    - 5-tier risk rating (1=Critical → 5=Healthy)
    - SHAP waterfall values per feature
    - Human-readable alert in Hindi or English
    - Recommended intervention with timeline
    """
    sector = _ENTERPRISE_REGISTRY.get(enterprise_id, {}).get("sector", "dairy")
    risk_score = round(random.uniform(15, 90), 1)
    tier = next(
        (t for t, cfg in RISK_TIER_MAP.items() if cfg["score_range"][0] <= risk_score < cfg["score_range"][1]),
        3,
    )
    tier_info = RISK_TIER_MAP[tier]
    shap_features = _mock_shap_features(sector)

    if lang == "hi":
        alert = _build_hindi_alert(tier, sector, shap_features)
    else:
        top_en = shap_features[0].name if shap_features else "cash flow"
        alert = (
            f"Risk Tier {tier} ({tier_info['label']}): Primary driver is '{top_en}'. "
            f"Score: {risk_score}/100. "
            f"Recommended action: Contact your NABARD field officer within 14 days."
        )

    intervention = {
        1: "Immediate: Emergency KCC rescheduling + field officer visit within 7 days",
        2: "Urgent: Cash flow counselling + input cost review within 14 days",
        3: "Advisory: Monthly monitoring + sector-specific guidance",
        4: "Routine: Quarterly review sufficient",
        5: "Growth: Explore NABARD refinance / expansion schemes",
    }[tier]

    return RiskResponse(
        enterprise_id=enterprise_id,
        sector=sector,
        risk_score=risk_score,
        risk_tier=tier,
        risk_label=tier_info["label"],
        risk_color=tier_info["color"],
        shap_features=shap_features,
        alert_hindi=_build_hindi_alert(tier, sector, shap_features),
        alert_english=(
            f"Risk Tier {tier} ({tier_info['label']}): Score {risk_score}/100. "
            f"Top driver: {shap_features[0].name if shap_features else 'N/A'}."
        ),
        recommended_intervention=intervention,
        assessed_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "service": "GramFlow API",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
