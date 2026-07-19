"""
Generate realistic synthetic mock data for GramFlow demo.
Run: python generate_mock_data.py
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "data" / "mock"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2025, 7, 1)
END_DATE = date(2026, 1, 28)   # ~180 days
ENTERPRISE_COUNT = 4


def date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def seasonal_factor(d: date, peak_months=(10, 11, 12), trough_months=(5, 6)):
    """Return a multiplier 0.7–1.3 based on month."""
    if d.month in peak_months:
        return random.uniform(1.10, 1.30)
    if d.month in trough_months:
        return random.uniform(0.70, 0.90)
    return random.uniform(0.95, 1.10)


# ── 1. dairy_enterprises.csv ─────────────────────────────────────────────────

def gen_dairy():
    rows = []
    enterprise_bases = {
        f"ENT-DAI-{i+1:03d}": {
            "yield_base": random.uniform(20, 60),        # liters/day
            "price_base": random.uniform(35, 50),        # ₹/liter
            "feed_base":  random.uniform(400, 900),      # ₹/day
            "upi_base":   random.randint(5, 30),
        }
        for i in range(ENTERPRISE_COUNT)
    }

    for eid, base in enterprise_bases.items():
        rolling_upi = []
        for d in date_range(START_DATE, END_DATE):
            sf = seasonal_factor(d)
            rain = max(0, random.gauss(5 if 6 <= d.month <= 9 else 1, 3))
            milk_yield = max(5, base["yield_base"] * sf + random.gauss(0, 2))
            milk_price = max(28, base["price_base"] + random.gauss(0, 2) - (rain * 0.1))
            feed_cost  = base["feed_base"] * (1 + (d.month - 1) / 24 * 0.1) + random.gauss(0, 50)
            upi_cnt    = max(0, int(base["upi_base"] * sf + random.gauss(0, 3)))
            upi_amt    = upi_cnt * random.uniform(80, 400)
            rolling_upi.append(upi_amt)
            upi_vel_7d = sum(rolling_upi[-7:]) / min(7, len(rolling_upi))
            revenue    = milk_yield * milk_price
            net_cf     = revenue - feed_cost + upi_amt * 0.15

            rows.append({
                "enterprise_id": eid,
                "date": d.isoformat(),
                "milk_yield_liters": round(milk_yield, 2),
                "milk_price_per_liter": round(milk_price, 2),
                "feed_cost_inr": round(feed_cost, 2),
                "upi_txn_count": upi_cnt,
                "upi_amount_inr": round(upi_amt, 2),
                "upi_velocity_7d": round(upi_vel_7d, 2),
                "rainfall_mm": round(rain, 2),
                "temperature_c": round(random.uniform(18, 38), 1),
                "gross_revenue_inr": round(revenue, 2),
                "net_cashflow_inr": round(net_cf, 2),
            })

    out = OUTPUT_DIR / "dairy_enterprises.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] dairy_enterprises.csv  ({len(rows)} rows)")


# ── 2. poultry_enterprises.csv ───────────────────────────────────────────────

def gen_poultry():
    rows = []
    enterprise_bases = {
        f"ENT-PLT-{i+1:03d}": {
            "bird_base":      random.randint(500, 2000),
            "egg_price_base": random.uniform(5.5, 8.5),   # ₹/egg
            "feed_base":      random.uniform(800, 2500),
            "mortality_base": random.uniform(0.002, 0.01),
            "upi_base":       random.randint(10, 50),
        }
        for i in range(ENTERPRISE_COUNT)
    }

    for eid, base in enterprise_bases.items():
        rolling_upi = []
        for d in date_range(START_DATE, END_DATE):
            sf = seasonal_factor(d, peak_months=(11, 12, 1), trough_months=(6, 7))
            birds      = int(base["bird_base"] * (1 - base["mortality_base"] * 30 * 0.5))
            egg_yield  = max(0, int(birds * 0.85 * sf + random.gauss(0, 20)))
            egg_price  = max(4, base["egg_price_base"] * sf + random.gauss(0, 0.3))
            mortality  = max(0, base["mortality_base"] + random.gauss(0, 0.001))
            feed_cost  = base["feed_base"] * sf + random.gauss(0, 100)
            upi_cnt    = max(0, int(base["upi_base"] * sf + random.gauss(0, 5)))
            upi_amt    = upi_cnt * random.uniform(60, 300)
            rolling_upi.append(upi_amt)
            upi_vel_7d = sum(rolling_upi[-7:]) / min(7, len(rolling_upi))
            revenue    = egg_yield * egg_price
            net_cf     = revenue - feed_cost

            rows.append({
                "enterprise_id": eid,
                "date": d.isoformat(),
                "bird_count": birds,
                "egg_yield_count": egg_yield,
                "egg_price_inr": round(egg_price, 2),
                "mortality_rate": round(mortality, 5),
                "feed_cost_inr": round(feed_cost, 2),
                "upi_txn_count": upi_cnt,
                "upi_amount_inr": round(upi_amt, 2),
                "upi_velocity_7d": round(upi_vel_7d, 2),
                "gross_revenue_inr": round(revenue, 2),
                "net_cashflow_inr": round(net_cf, 2),
            })

    out = OUTPUT_DIR / "poultry_enterprises.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] poultry_enterprises.csv ({len(rows)} rows)")


# ── 3. food_processing_enterprises.csv ───────────────────────────────────────

def gen_food_processing():
    rows = []
    products = ["pickles", "papad", "masala_powder", "dried_mango"]
    enterprise_bases = {
        f"ENT-FPR-{i+1:03d}": {
            "volume_base":    random.uniform(50, 300),     # kg/day processed
            "sale_price":     random.uniform(60, 180),     # ₹/kg
            "raw_mat_ratio":  random.uniform(0.35, 0.55),
            "labor_base":     random.uniform(500, 1500),
            "upi_base":       random.randint(8, 40),
            "product":        random.choice(products),
        }
        for i in range(ENTERPRISE_COUNT)
    }

    for eid, base in enterprise_bases.items():
        rolling_upi = []
        for d in date_range(START_DATE, END_DATE):
            sf = seasonal_factor(d, peak_months=(10, 2, 3), trough_months=(7, 8))
            volume    = max(5, base["volume_base"] * sf + random.gauss(0, 15))
            sale_price = base["sale_price"] * (1 + 0.003 * (d - START_DATE).days / 30) + random.gauss(0, 5)
            raw_mat   = volume * sale_price * base["raw_mat_ratio"]
            labor     = base["labor_base"] + random.gauss(0, 100)
            upi_cnt   = max(0, int(base["upi_base"] * sf + random.gauss(0, 4)))
            upi_amt   = upi_cnt * random.uniform(100, 600)
            rolling_upi.append(upi_amt)
            upi_vel_7d = sum(rolling_upi[-7:]) / min(7, len(rolling_upi))
            revenue   = volume * sale_price
            net_cf    = revenue - raw_mat - labor

            rows.append({
                "enterprise_id": eid,
                "date": d.isoformat(),
                "product_type": base["product"],
                "processing_volume_kg": round(volume, 2),
                "sale_price_per_kg_inr": round(sale_price, 2),
                "raw_material_cost_inr": round(raw_mat, 2),
                "labor_cost_inr": round(labor, 2),
                "upi_txn_count": upi_cnt,
                "upi_amount_inr": round(upi_amt, 2),
                "upi_velocity_7d": round(upi_vel_7d, 2),
                "gross_revenue_inr": round(revenue, 2),
                "net_cashflow_inr": round(net_cf, 2),
            })

    out = OUTPUT_DIR / "food_processing_enterprises.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] food_processing_enterprises.csv ({len(rows)} rows)")


# ── 4. handicrafts_enterprises.csv ───────────────────────────────────────────

def gen_handicrafts():
    rows = []
    craft_types = ["bamboo_weaving", "pottery", "block_printing", "jute_bags"]
    enterprise_bases = {
        f"ENT-HND-{i+1:03d}": {
            "order_base":      random.uniform(3, 15),     # orders/day
            "avg_order_value": random.uniform(500, 3000),
            "material_ratio":  random.uniform(0.25, 0.40),
            "labor_base":      random.uniform(300, 1200),
            "upi_base":        random.randint(3, 20),
            "craft":           random.choice(craft_types),
        }
        for i in range(ENTERPRISE_COUNT)
    }

    for eid, base in enterprise_bases.items():
        rolling_upi = []
        for d in date_range(START_DATE, END_DATE):
            # Festival seasons boost handicrafts significantly
            festival_boost = 1.5 if d.month in (10, 11) else 1.0
            sf = seasonal_factor(d, peak_months=(10, 11, 12), trough_months=(1, 2)) * festival_boost
            orders    = max(0, base["order_base"] * sf + random.gauss(0, 1))
            avg_val   = base["avg_order_value"] * (1 + (d.month in (10, 11)) * 0.3) + random.gauss(0, 100)
            material  = orders * avg_val * base["material_ratio"]
            labor     = base["labor_base"] * min(orders / base["order_base"], 2.0) + random.gauss(0, 80)
            skilled_days = max(0, round(orders * 0.8 + random.gauss(0, 0.5), 1))
            upi_cnt   = max(0, int(base["upi_base"] * sf + random.gauss(0, 2)))
            upi_amt   = upi_cnt * random.uniform(150, 1000)
            rolling_upi.append(upi_amt)
            upi_vel_7d = sum(rolling_upi[-7:]) / min(7, len(rolling_upi))
            revenue   = orders * avg_val
            net_cf    = revenue - material - labor

            rows.append({
                "enterprise_id": eid,
                "date": d.isoformat(),
                "craft_type": base["craft"],
                "orders_received": round(orders, 1),
                "avg_order_value_inr": round(avg_val, 2),
                "skilled_labor_days": skilled_days,
                "material_cost_inr": round(material, 2),
                "labor_cost_inr": round(labor, 2),
                "upi_txn_count": upi_cnt,
                "upi_amount_inr": round(upi_amt, 2),
                "upi_velocity_7d": round(upi_vel_7d, 2),
                "gross_revenue_inr": round(revenue, 2),
                "net_cashflow_inr": round(net_cf, 2),
            })

    out = OUTPUT_DIR / "handicrafts_enterprises.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] handicrafts_enterprises.csv ({len(rows)} rows)")


if __name__ == "__main__":
    print("Generating GramFlow mock datasets...\n")
    gen_dairy()
    gen_poultry()
    gen_food_processing()
    gen_handicrafts()
    print(f"\nAll files written to: {OUTPUT_DIR.resolve()}")
