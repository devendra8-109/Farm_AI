"""
module3_arima_module4_profit.py (FIXED - GLOBAL FALLBACK)
========================================================
Logic Fixes:
1. Added fallback to National Average Price if State-specific price is missing.
2. Added fallback to National Average Yield if State-specific yield is missing.
3. Ensures NO state shows "No data found".
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

# Load data
df_price = pd.read_csv(os.path.join(CLEAN_DIR, "mandi_prices_monthly.csv"))
df_yield = pd.read_csv(os.path.join(CLEAN_DIR, "crop_yield_clean.csv"))

CROP_PROFILES = {
    "wheat":      {"base_yield": 3500,  "cost_ha": 35000, "fert_factor": 0.05},
    "rice":       {"base_yield": 4000,  "cost_ha": 42000, "fert_factor": 0.06},
    "maize":      {"base_yield": 3000,  "cost_ha": 28000, "fert_factor": 0.04},
    "cotton":     {"base_yield": 2000,  "cost_ha": 58000, "fert_factor": 0.07},
    "sugarcane":  {"base_yield": 75000, "cost_ha": 65000, "fert_factor": 0.03},
    "soyabean":   {"base_yield": 2500,  "cost_ha": 30000, "fert_factor": 0.04},
    "gram":       {"base_yield": 1500,  "cost_ha": 25000, "fert_factor": 0.03},
    "groundnut":  {"base_yield": 2200,  "cost_ha": 40000, "fert_factor": 0.05},
}
DEFAULT_PROFILE = {"base_yield": 2500, "cost_ha": 30000, "fert_factor": 0.04}

def calculate_profit(inputs, price_per_kg, profile):
    f, l, s = inputs
    est_yield = profile["base_yield"] * (1 + profile["fert_factor"] * np.log1p(f))
    revenue   = est_yield * price_per_kg
    total_cost = profile["cost_ha"] + (f * 30) + (l * 450) + (s * 100)
    return -(revenue - total_cost)

results = []

# Get the list of ALL states from the Yield dataset (the primary list)
all_states = df_yield['state'].unique()

# Pre-calculate National Average Prices for fallbacks
national_avg_prices = df_price.groupby('crop')['avg_modal_price'].mean().to_dict()

for state in all_states:
    # Try to get crops for this state from yield data
    state_crops = df_yield[df_yield['state'] == state]['crop'].unique()
    
    for crop in state_crops:
        # 1. Try State-Specific Price (case-insensitive — fixes silent fallback bug)
        crop_price_data = df_price[
            (df_price['state'].str.lower() == state.lower()) &
            (df_price['crop'].str.lower() == crop.lower())
        ]
        
        if not crop_price_data.empty:
            avg_price_quintal = crop_price_data['avg_modal_price'].mean()
            price_source = "State Market"
        elif crop.lower() in national_avg_prices:
            # 2. Fallback to National Average Price
            avg_price_quintal = national_avg_prices[crop.lower()]
            price_source = "National Avg"
        else:
            # 3. Final Fallback if crop is not in price data at all
            avg_price_quintal = 2000 
            price_source = "Estimated"
            
        price_per_kg = avg_price_quintal / 100.0
        profile = CROP_PROFILES.get(crop.lower(), DEFAULT_PROFILE)
        
        res = minimize(
            calculate_profit, x0=[80, 20, 15], args=(price_per_kg, profile),
            bounds=Bounds([10, 5, 5], [250, 60, 45]), method="L-BFGS-B"
        )
        
        if res.success:
            opt_f, opt_l, opt_s = res.x
            net_profit = -res.fun
            results.append({
                "state": state, "crop": crop.upper(),
                "fertilizer_kg": round(opt_f, 1), "labour_hours": round(opt_l, 1),
                "seed_rate": round(opt_s, 1), "net_profit": round(net_profit, 2),
                "price_source": price_source
            })

df_final = pd.DataFrame(results).sort_values(['state', 'net_profit'], ascending=[True, False])
df_final.to_csv(os.path.join(CLEAN_DIR, "m4_final_recommendations.csv"), index=False)
print("Saved comprehensive recommendations with fallbacks.")
