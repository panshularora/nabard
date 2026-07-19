"""
Pydantic request/response models for the GramFlow API.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------


class OnboardRequest(BaseModel):
    enterprise_name: str = Field(..., example="Sharma Dairy Farm", description="Name of the micro-enterprise")
    sector: str = Field(
        ...,
        example="dairy",
        description="Sector: dairy | poultry | food_processing | handicrafts | rural_retail",
    )
    district: str = Field(..., example="Nashik", description="District where enterprise operates")
    state: str = Field(..., example="Maharashtra", description="State")
    owner_name: str = Field(..., example="Ramesh Sharma")
    owner_mobile: Optional[str] = Field(None, example="9876543210")
    annual_turnover_inr: Optional[float] = Field(None, example=350000.0)


class OnboardResponse(BaseModel):
    enterprise_id: str
    sector: str
    risk_weights: Dict[str, float]
    seasonal_peaks: List[str]
    seasonal_troughs: List[str]
    key_inputs: List[str]
    message: str


# ---------------------------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    enterprise_id: str = Field(..., example="ENT-DAI-A1B2C3D4")
    date_range_start: Optional[date] = Field(None, description="Start of data pull window")
    date_range_end: Optional[date] = Field(None, description="End of data pull window")
    sources: Optional[List[str]] = Field(
        default=["commodity_prices", "climate", "upi", "mobile_logs"],
        description="Data sources to pull",
    )


class IngestResponse(BaseModel):
    job_id: str
    enterprise_id: str
    status: str  # queued | running | completed | failed
    records_ingested: Dict[str, int]
    ingestion_date: date
    message: str


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


class ForecastPoint(BaseModel):
    ds: date = Field(..., description="Forecast date")
    yhat: float = Field(..., description="Point forecast (INR)")
    yhat_lower: float = Field(..., description="Lower confidence bound (INR)")
    yhat_upper: float = Field(..., description="Upper confidence bound (INR)")


class ForecastResponse(BaseModel):
    enterprise_id: str
    sector: str
    horizon_months: int
    model: str
    forecast: List[ForecastPoint]
    total_projected_cashflow: float
    trend_direction: str  # upward | downward | flat
    generated_at: datetime
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Risk Intelligence
# ---------------------------------------------------------------------------


class ShapFeature(BaseModel):
    name: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value (normalised)")
    shap_value: float = Field(..., description="SHAP attribution value")


class RiskResponse(BaseModel):
    enterprise_id: str
    sector: str
    risk_score: float = Field(..., ge=0, le=100, description="Composite risk score (0=worst, 100=best)")
    risk_tier: int = Field(..., ge=1, le=5, description="Risk tier: 1=Critical, 5=Healthy")
    risk_label: str
    risk_color: str  # hex color for UI rendering
    shap_features: List[ShapFeature]
    alert_hindi: str
    alert_english: str
    recommended_intervention: str
    assessed_at: datetime
