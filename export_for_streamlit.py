# ---- Paste this as the LAST cell of World-Development.ipynb and run it ----
import os, json
import numpy as np

os.makedirs("streamlit_app/data", exist_ok=True)

features = list(df_scaled.columns)                      # the 22 model features
model_ready = df_clustering[features].copy()            # country-level, log-transformed (what the scaler saw)
model_ready.to_csv("streamlit_app/data/country_features.csv")   # index name = Country

meta = {"features": features, "log_cols": [str(c) for c in transform_cols]}
with open("streamlit_app/data/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

final_model_comparison.to_csv("streamlit_app/data/model_comparison.csv", index=False)
print("Saved:", os.listdir("streamlit_app/data"))
