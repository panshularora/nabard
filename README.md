<div align="center">

<img src="docs/architecture.png" alt="GramFlow Architecture" width="100%"/>

# GramFlow — AI Cash Flow Intelligence for Rural MSMEs
### NABARD × GramFlow | Round 1 Submission

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NABARD](https://img.shields.io/badge/Partner-NABARD-003087?logoColor=white)](https://nabard.org)

> **GramFlow** empowers NABARD field officers and rural micro-enterprise owners with AI-driven cash-flow forecasting, 5-tier risk classification, and explainable alerts in Hindi — all designed to work offline in low-connectivity rural environments.

</div>

---

## 🏗️ Architecture Overview

GramFlow implements an **8-stage intelligence pipeline** connecting raw rural enterprise data to actionable interventions:

```
Enterprise Onboarding
        ↓
Daily Data Ingestion (Airflow)
   ├── Commodity prices (AGMARKNET / eNAM)
   ├── Climate signals (IMD API)
   ├── UPI aggregate proxies
   └── Mobile income/expense logs
        ↓
Feature Engineering
   ├── Rolling UPI velocity (7d, 14d, 30d)
   ├── Commodity price lag features
   ├── Seasonal dummies + harvest cycle alignment
   └── Sector-specific input cost ratios
        ↓
Forecast Generation
   ├── NeuralProphet (trend + seasonality decomposition)
   └── LightGBM (3–6 month cash flow + confidence intervals)
        ↓
Risk Classification
   ├── XGBoost (5-tier risk rating)
   └── SHAP (feature attribution → Hindi/English alert)
        ↓
Delivery Layer
   ├── Field Officer PWA (portfolio risk panel + prioritized list)
   └── Enterprise Owner View (forecast + specific action)
        ↓
Offline Operation
   ├── Service Worker (UI cache)
   ├── IndexedDB (local enterprise data + forecasts)
   └── ONNX Runtime (in-browser risk scoring)
        ↓
Feedback Loop
   └── Intervention outcomes → Quarterly model retraining
```

---

## 🗂️ Sector Coverage

| Sector | Enterprises | Key Risk Signals | Seasonal Peak | Seasonal Trough |
|--------|-------------|-----------------|---------------|-----------------|
| 🐄 **Dairy** | Dairy farms, milk cooperatives | Milk price volatility, feed cost ratio, rainfall index | Oct–Dec | Apr–May |
| 🐔 **Poultry** | Broiler/layer farms | Egg price volatility, mortality rate, feed cost | Nov–Jan | Jun–Jul |
| 🏭 **Food Processing** | Pickle, papad, masala, dried fruit units | Raw material cost, sale price index, labor cost | Oct, Feb–Mar | Jul–Aug |
| 🧶 **Handicrafts** | Bamboo weaving, pottery, block printing, jute bags | Order velocity, material cost, festival demand | Oct–Dec (Diwali) | Jan–Feb |
| 🏪 **Rural Retail** | Village kiranas, agri-input shops | UPI velocity, inventory turnover, credit utilization | Oct–Nov | Jun–Jul |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | Apache Airflow 2.x | Daily ETL pipeline scheduling |
| **API** | FastAPI + Uvicorn | REST endpoints, async request handling |
| **Forecasting** | NeuralProphet | Seasonal decomposition, trend detection |
| **ML — Cash Flow** | LightGBM | Gradient-boosted 3–6 month projection |
| **ML — Risk** | XGBoost | 5-tier risk classification |
| **Explainability** | SHAP | Feature attribution, Hindi alert generation |
| **Frontend** | React PWA | Field officer dashboard, offline-capable |
| **Offline Inference** | ONNX Runtime (in-browser) | Risk scoring without connectivity |
| **Offline Storage** | IndexedDB + Service Worker | Local enterprise data + forecast cache |
| **Data** | Pandas, NumPy | Feature engineering pipeline |
| **Visualization** | Matplotlib, Plotly | Forecast + SHAP charts |
| **Validation** | Pydantic v2 | Request/response schema enforcement |

---

## 📊 Sample SHAP Output — Hindi Alert

Below is a real output from the GramFlow risk engine for a dairy enterprise in Maharashtra:

```
===========================================================
GRAMFLOW जोखिम चेतावनी — NABARD
===========================================================
उद्यम ID    : ENT-DAI-001
क्षेत्र      : डेयरी
जोखिम स्तर  : 🟠 उच्च जोखिम (Tier 2/5)
जोखिम स्कोर : 83%
-----------------------------------------------------------
मुख्य जोखिम कारण :  UPI लेनदेन गति (7 दिन) और पिछले सप्ताह का नकदी प्रवाह
सुरक्षात्मक कारक : चारे की लागत अनुपात और दूध मूल्य प्रति लीटर
अनुशंसित कार्य    : पशु बीमा और KCC ऋण की समीक्षा करें
समय-सीमा        : अगले 14 दिनों में NABARD फील्ड अधिकारी से मिलें
===========================================================
```

### SHAP Waterfall

![SHAP Waterfall](docs/shap_waterfall.png)

### 6-Month Cash Flow Forecast

![Forecast Plot](docs/forecast_plot.png)

---

## 📁 Repository Structure

```
nabard/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── generate_mock_data.py              # Mock data generator
├── build_notebook.py                  # Notebook builder
│
├── api/                               # FastAPI backend
│   ├── __init__.py
│   ├── main.py                        # 4 core endpoints
│   └── models.py                      # Pydantic schemas
│
├── data/
│   └── mock/                          # Synthetic enterprise data
│       ├── dairy_enterprises.csv      # 848 rows × 4 enterprises
│       ├── poultry_enterprises.csv    # 848 rows × 4 enterprises
│       ├── food_processing_enterprises.csv
│       └── handicrafts_enterprises.csv
│
├── notebooks/
│   └── gramflow_demo.ipynb            # Full pipeline demo (pre-executed)
│
└── docs/
    ├── architecture.png               # System architecture diagram
    ├── forecast_plot.png              # Prophet forecast output
    └── shap_waterfall.png             # SHAP attribution chart
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/panshularora/nabard.git
cd nabard
pip install -r requirements.txt
```

### 2. Generate Mock Data

```bash
python generate_mock_data.py
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at → **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 4. Run the Demo Notebook

```bash
jupyter notebook notebooks/gramflow_demo.ipynb
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/onboard` | Register enterprise, load sector template |
| `POST` | `/v1/ingest` | Trigger daily data ingestion pipeline |
| `GET`  | `/v1/forecast/{enterprise_id}` | 6-month cash flow forecast |
| `GET`  | `/v1/risk/{enterprise_id}` | 5-tier risk + SHAP Hindi alert |
| `GET`  | `/health` | API health check |

### Sample: Onboard an Enterprise

```bash
curl -X POST http://localhost:8000/v1/onboard \
  -H "Content-Type: application/json" \
  -d '{
    "enterprise_name": "Sharma Dairy Farm",
    "sector": "dairy",
    "district": "Nashik",
    "state": "Maharashtra",
    "owner_name": "Ramesh Sharma",
    "owner_mobile": "9876543210"
  }'
```

**Response:**
```json
{
  "enterprise_id": "ENT-DAI-A1B2C3D4",
  "sector": "dairy",
  "risk_weights": {
    "upi_velocity": 0.30,
    "milk_price_volatility": 0.25,
    "feed_cost_ratio": 0.25,
    "rainfall_index": 0.20
  },
  "seasonal_peaks": ["October", "November", "December"],
  "seasonal_troughs": ["April", "May"],
  "key_inputs": ["cattle_feed", "veterinary_services", "electricity"],
  "message": "Enterprise 'Sharma Dairy Farm' successfully registered."
}
```

### Sample: Get Risk Rating + Hindi Alert

```bash
curl http://localhost:8000/v1/risk/ENT-DAI-A1B2C3D4?lang=hi
```

---

## 📈 Risk Tier System

| Tier | Label | Score Range | Recommended Action |
|------|-------|-------------|-------------------|
| 1 🔴 | Critical | 0–20 | Emergency KCC rescheduling + field visit within 7 days |
| 2 🟠 | High | 20–40 | Cash flow counselling + input cost review within 14 days |
| 3 🟡 | Moderate | 40–60 | Monthly monitoring + sector guidance |
| 4 🟢 | Low | 60–80 | Quarterly review |
| 5 ✅ | Healthy | 80–100 | Explore NABARD expansion schemes |

---

## 🌐 Offline Architecture

GramFlow is designed for **Bharat-first connectivity**:

- **Service Worker**: Caches entire PWA shell for offline use
- **IndexedDB**: Stores enterprise data and last 3 forecast snapshots locally
- **ONNX Runtime (WASM)**: Risk scoring model runs entirely in-browser — no server needed
- **Background Sync API**: Queues all offline writes, syncs automatically when connectivity resumes

---

## 🔄 Feedback Loop

Field officers log intervention outcomes (loan restructured, advisory given, no action needed) via the PWA. These outcomes are:

1. Tagged to enterprise + risk tier + features at time of alert
2. Aggregated quarterly into a retraining dataset
3. Used to fine-tune XGBoost risk thresholds and LightGBM forecast weights

This closes the loop from **AI alert → real-world intervention → model improvement**.

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
  <strong>Built for NABARD × Rural Innovation Challenge 2026</strong><br/>
  Empowering 63 million rural MSMEs with AI-driven financial intelligence
</div>
