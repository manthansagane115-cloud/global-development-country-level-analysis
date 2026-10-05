import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="World Development Clusters", page_icon="🌍", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
RANK_COL = "Life Expectancy Female"  # clusters are ordered low -> high by this feature
PALETTE = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4"]
NOISE_COLOR = "#9e9e9e"


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    X_log = pd.read_csv(DATA_DIR / "country_features.csv", index_col="Country")
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    features, log_cols = meta["features"], meta["log_cols"]
    X_log = X_log[features]

    # Back to original units (undo the natural log used in the notebook)
    X_raw = X_log.copy()
    X_raw[log_cols] = np.exp(X_raw[log_cols])

    scaler = StandardScaler().fit(X_log)
    X_scaled = scaler.transform(X_log)

    pca = PCA(n_components=2).fit(X_scaled)
    coords = pca.transform(X_scaled)

    comp_path = DATA_DIR / "model_comparison.csv"
    comparison = pd.read_csv(comp_path) if comp_path.exists() else None

    return {
        "X_log": X_log, "X_raw": X_raw, "features": features, "log_cols": log_cols,
        "scaler": scaler, "X_scaled": X_scaled, "pca": pca, "coords": coords,
        "comparison": comparison,
    }


# ----------------------------------------------------------------------------
# Modelling helpers
# ----------------------------------------------------------------------------
def fit_model(algo, p, X):
    """Returns (model_or_None, labels). Same settings as the notebook's tuned models."""
    if algo == "K-Means":
        model = KMeans(n_clusters=p["k"], random_state=42, n_init=10).fit(X)
        return model, model.labels_
    if algo == "Gaussian Mixture (GMM)":
        model = GaussianMixture(n_components=p["k"], covariance_type=p["cov"], random_state=42).fit(X)
        return model, model.predict(X)
    if algo == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=p["k"], linkage=p["linkage"]).fit_predict(X)
        return None, labels
    labels = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit_predict(X)
    return None, labels


def reorder_labels(labels, rank_values):
    """Renumber clusters so 0 = lowest mean life expectancy ... k-1 = highest. Noise stays -1."""
    clusters = [c for c in np.unique(labels) if c != -1]
    order = sorted(clusters, key=lambda c: rank_values[labels == c].mean())
    mapping = {old: new for new, old in enumerate(order)}
    mapping[-1] = -1
    return np.array([mapping[l] for l in labels]), mapping


def cluster_name(c, k):
    if c == -1:
        return "Noise / outlier"
    if k == 2:
        return ["Lower development indicators", "Higher development indicators"][c]
    return f"Tier {c + 1} of {k}"


def compute_metrics(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return None
    return {
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X[mask], labels[mask]),
    }


@st.cache_data
def kmeans_sweep(X):
    rows = []
    for k in range(2, 11):
        lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rows.append({"K": k, "Inertia": lab.inertia_, "Silhouette": silhouette_score(X, lab.labels_)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Load + sidebar
# ----------------------------------------------------------------------------
try:
    A = load_artifacts()
except FileNotFoundError:
    st.error("Missing files in `data/`. Run the export cell at the end of the notebook first "
             "(it creates country_features.csv and meta.json).")
    st.stop()

features, log_cols = A["features"], A["log_cols"]
X_raw, X_scaled, coords = A["X_raw"], A["X_scaled"], A["coords"]

st.sidebar.title("⚙️ Model settings")
algo = st.sidebar.selectbox("Clustering algorithm",
                            ["K-Means", "Gaussian Mixture (GMM)", "Agglomerative", "DBSCAN"])
params = {}
if algo == "K-Means":
    params["k"] = st.sidebar.slider("Number of clusters (K)", 2, 8, 2)
elif algo == "Gaussian Mixture (GMM)":
    params["k"] = st.sidebar.slider("Number of components", 2, 8, 2)
    params["cov"] = st.sidebar.selectbox("Covariance type", ["spherical", "full", "tied", "diag"])
elif algo == "Agglomerative":
    params["k"] = st.sidebar.slider("Number of clusters", 2, 8, 2)
    params["linkage"] = st.sidebar.selectbox("Linkage", ["complete", "ward", "average"])
else:
    params["eps"] = st.sidebar.slider("eps", 1.0, 5.0, 2.75, 0.25)
    params["min_samples"] = st.sidebar.slider("min_samples", 3, 10, 5)
st.sidebar.caption("Defaults = the tuned settings from the notebook (K=2 for most models; "
                   "DBSCAN eps=2.75, min_samples=5).")

model, raw_labels = fit_model(algo, params, X_scaled)
labels, mapping = reorder_labels(raw_labels, X_raw[RANK_COL].values)
k_found = len(set(labels) - {-1})
names = {c: cluster_name(c, k_found) for c in set(labels)}
color_map = {names[c]: (NOISE_COLOR if c == -1 else PALETTE[c % len(PALETTE)]) for c in set(labels)}
order_names = [names[c] for c in sorted(set(labels))]

result = pd.DataFrame({"Country": X_raw.index, "Group": [names[l] for l in labels]}, index=X_raw.index)
metrics = compute_metrics(X_scaled, labels)

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("🌍 Global Development Clustering")
st.caption("Grouping 208 countries by economic, health, technology and demographic indicators.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clusters found", k_found)
c2.metric("Noise points", int((labels == -1).sum()))
if metrics:
    c3.metric("Silhouette ↑", f"{metrics['Silhouette']:.3f}")
    c4.metric("Davies-Bouldin ↓", f"{metrics['Davies-Bouldin']:.3f}")
    c5.metric("Calinski-Harabasz ↑", f"{metrics['Calinski-Harabasz']:.1f}")
else:
    st.warning("This setting produced fewer than 2 clusters, so metrics can't be computed. Change the parameters.")

tab_about, tab_explore, tab_lookup, tab_predict, tab_compare = st.tabs(
    ["📖 About", "🗺️ Cluster explorer", "🔎 Country lookup", "🧮 Classify a country", "📊 Model comparison"]
)

# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**Business objective:** create clusters on a global development measurement dataset, compare several
clustering models, and deploy the result as an application.

**Pipeline used in the notebook**
1. Cleaned the data (dropped *Ease of Business*, ~93% missing; converted `%` and `$` strings to numbers).
2. Imputed missing values with the country median, then the global median.
3. Log-transformed features with |skewness| > 2 to tame extreme values (GDP, CO₂, population, tourism ...).
4. Aggregated 13 yearly records per country into one median row per country (208 rows).
5. Standardised the features, then tuned K-Means, Agglomerative, GMM and DBSCAN.
6. Compared models using Silhouette, Davies-Bouldin and Calinski-Harabasz scores plus cluster sizes.

**How to use this app:** pick an algorithm in the sidebar, then explore the map and cluster profiles,
look up a single country, or enter indicator values to see which group a country would fall into.
"""
    )
    st.info("Clusters are numbered from lowest to highest average female life expectancy, "
            "so the colours and names stay consistent when you change settings.")

# ----------------------------------------------------------------------------
# Explorer
# ----------------------------------------------------------------------------
with tab_explore:
    left, right = st.columns(2)
    with left:
        st.subheader("World map")
        fig = px.choropleth(result, locations="Country", locationmode="country names",
                            color="Group", hover_name="Country",
                            color_discrete_map=color_map, category_orders={"Group": order_names})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
        st.plotly_chart(fig)
        st.caption("Countries whose names don't match the map's naming are simply not drawn.")
    with right:
        st.subheader("PCA view (2 components)")
        pdf = result.assign(PC1=coords[:, 0], PC2=coords[:, 1])
        fig = px.scatter(pdf, x="PC1", y="PC2", color="Group", hover_name="Country",
                         color_discrete_map=color_map, category_orders={"Group": order_names})
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig)
        st.caption(f"The two components explain {A['pca'].explained_variance_ratio_.sum():.0%} of the variance.")

    st.subheader("Cluster sizes")
    sizes = result["Group"].value_counts().reindex(order_names).rename("Countries").to_frame()
    st.dataframe(sizes)

    st.subheader("What separates the clusters? (standardised averages)")
    z = pd.DataFrame(X_scaled, index=X_raw.index, columns=features)
    z["Group"] = result["Group"]
    z_profile = z.groupby("Group").mean().reindex(order_names).T
    fig = px.imshow(z_profile, aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2)
    fig.update_layout(coloraxis_colorbar_title="z-score")
    st.plotly_chart(fig)

    st.subheader("Cluster profile (median, original units)")
    profile = X_raw.assign(Group=result["Group"]).groupby("Group").median().reindex(order_names).T
    st.dataframe(profile.round(2))

    st.subheader("Countries in each cluster")
    pick = st.selectbox("Choose a group", order_names)
    st.write(", ".join(result.index[result["Group"] == pick]))
    st.download_button("⬇️ Download all assignments (CSV)",
                       result.reset_index(drop=True).to_csv(index=False).encode(),
                       "country_clusters.csv", "text/csv")

# ----------------------------------------------------------------------------
# Country lookup
# ----------------------------------------------------------------------------
with tab_lookup:
    country = st.selectbox("Country", list(X_raw.index))
    grp = result.loc[country, "Group"]
    st.markdown(f"### {country} → **{grp}**")
    same = X_raw[result["Group"] == grp]
    table = pd.DataFrame({
        country: X_raw.loc[country],
        "Median of its cluster": same.median(),
        "Median of all countries": X_raw.median(),
    })
    st.dataframe(table.round(2))

    dist = np.linalg.norm(X_scaled - X_scaled[list(X_raw.index).index(country)], axis=1)
    nearest = pd.Series(dist, index=X_raw.index).drop(country).nsmallest(5)
    st.markdown("**Most similar countries:** " + ", ".join(nearest.index))

# ----------------------------------------------------------------------------
# Classify a new country
# ----------------------------------------------------------------------------
with tab_predict:
    if model is None:
        st.info(f"{algo} can't assign new points to clusters. Switch to **K-Means** or **GMM** in the sidebar.")
    else:
        st.write("Enter indicator values in **original units** (defaults = median of all countries).")
        defaults = X_raw.median()
        with st.form("predict_form"):
            cols = st.columns(3)
            vals = {}
            for i, f in enumerate(features):
                step = float(max(abs(defaults[f]) * 0.05, 0.01))
                vals[f] = cols[i % 3].number_input(f, min_value=0.0, value=float(defaults[f]),
                                                   step=step, format="%.3f")
            go = st.form_submit_button("Classify")
        if go:
            x = pd.DataFrame([vals])[features]
            x[log_cols] = np.log(np.clip(x[log_cols].astype(float), 1e-9, None))
            xs = A["scaler"].transform(x)
            old = int(model.predict(xs)[0])
            new = mapping[old]
            st.success(f"Predicted group: **{names[new]}**")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(xs)[0]
                st.write({names[mapping[o]]: f"{p:.1%}" for o, p in enumerate(proba)})
            d = np.linalg.norm(X_scaled - xs, axis=1)
            st.markdown("**Most similar countries:** " + ", ".join(pd.Series(d, index=X_raw.index).nsmallest(5).index))

# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------
with tab_compare:
    if A["comparison"] is not None:
        st.subheader("Tuned models from the notebook")
        st.dataframe(A["comparison"].round(4))
        fig = px.bar(A["comparison"], x="Model", y="Silhouette", color="Model", text_auto=".3f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        st.caption("DBSCAN's scores are computed after removing noise points (57 of 208 countries), "
                   "so they are not directly comparable with the other three models.")
    st.subheader("K-Means: choosing K")
    sweep = kmeans_sweep(X_scaled)
    a, b = st.columns(2)
    a.plotly_chart(px.line(sweep, x="K", y="Inertia", markers=True, title="Elbow (inertia)"))
    b.plotly_chart(px.line(sweep, x="K", y="Silhouette", markers=True, title="Silhouette score"))
