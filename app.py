import streamlit as st
import random, hashlib
from datetime import date
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Phineas & Ferb: Project Predictor (AI+)",
                   page_icon="🛠️",
                   layout="wide")

# ----------------------
# Cartoon-styled CSS (same vibe as before)
# ----------------------
CARTOON_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Comic+Neue:wght@300;400;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"]  { font-family: 'Comic Neue', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .cartoon-bg { background:
      radial-gradient(circle at 15% 20%, #fff7a1 0 20%, transparent 21% 100%),
      radial-gradient(circle at 85% 10%, #ffd6e7 0 18%, transparent 19% 100%),
      radial-gradient(circle at 75% 85%, #c4f2ff 0 22%, transparent 23% 100%),
      radial-gradient(circle at 10% 80%, #c3ffa8 0 18%, transparent 19% 100%),
      linear-gradient(180deg, #f2f7ff, #e9f5ff); min-height: 100vh; padding-bottom: 2rem; }
  .cartoon-card { background: #fff; border: 4px solid #111; border-radius: 18px; box-shadow: 6px 6px 0 #111; padding: 1.2rem; margin-bottom: 1rem; }
  .cartoon-title { display: inline-block; background: #ffe66d; border: 4px solid #111; border-radius: 16px; padding: 8px 14px; box-shadow: 6px 6px 0 #111; font-weight: 800; letter-spacing: 0.5px; }
  .badge { display: inline-block; background: #9bf6ff; border: 3px solid #111; border-radius: 999px; padding: 4px 10px; margin-right: 6px; margin-bottom: 6px; font-weight: 700; }
  .big-btn { background: linear-gradient(0deg, #ffb703, #ffd166); border: 4px solid #111; border-radius: 16px; padding: 12px 18px; box-shadow: 6px 6px 0 #111; font-weight: 900; text-transform: uppercase; letter-spacing: 0.8px; cursor: pointer; }
  .blueprint { background: repeating-linear-gradient(0deg, #0a2540 0px, #0a2540 2px, #0d335e 2px, #0d335e 34px); color: #c7e0ff; border: 4px solid #001d3d; border-radius: 14px; padding: 16px; font-family: "Courier New", monospace; box-shadow: 6px 6px 0 #001d3d; }
  .footer-note { opacity: 0.8; font-size: 0.9rem; }
</style>
"""
st.markdown('<div class="cartoon-bg">', unsafe_allow_html=True)
st.markdown(CARTOON_CSS, unsafe_allow_html=True)
st.markdown('<h1 class="cartoon-title">Phineas & Ferb: Project Predictor 🛠️🚀 — AI+ Mode</h1>', unsafe_allow_html=True)
st.write("")

# ----------------------
# Data & Projects
# ----------------------
DEFAULT_PROJECTS = [
    {"name": "Backyard Roller Coaster 2.0", "scale": 5, "tags": ["Wood", "Electronics"], "needs_space": True},
    {"name": "Time-Traveling Sandwich Maker", "scale": 3, "tags": ["Electronics", "Gadgets"], "needs_space": False},
    {"name": "Suburban Space Elevator", "scale": 6, "tags": ["Metal", "Electronics"], "needs_space": True},
    {"name": "Ocean-Wave Surf Simulator", "scale": 4, "tags": ["Rubber Bands", "Electronics"], "needs_space": False},
    {"name": "Doof-Inator Disruptor Shield", "scale": 2, "tags": ["Gadgets", "Electronics"], "needs_space": False},
    {"name": "Rainbow-Powered Jetpacks", "scale": 5, "tags": ["Jet Fuel (uh-oh)", "Metal"], "needs_space": False},
    {"name": "Giant Rube Goldberg Ice-Cream Machine", "scale": 4, "tags": ["Ice Cream", "Wood", "Gadgets"], "needs_space": True},
    {"name": "Underground Hyper-Tunnel", "scale": 6, "tags": ["Metal"], "needs_space": True},
    {"name": "DIY Snow-Globe City", "scale": 3, "tags": ["Paint", "Wood"], "needs_space": False},
    {"name": "Skywriting Drone Swarm", "scale": 4, "tags": ["Electronics"], "needs_space": False},
]

PROJECT_NAMES = [p["name"] for p in DEFAULT_PROJECTS]

MATERIALS_MASTER = ["Wood", "Metal", "Electronics", "Jet Fuel (uh-oh)", "Gadgets", "Paint", "Ice Cream", "Rubber Bands"]
CREW_MASTER = ["Candace", "Isabella", "Baljeet", "Buford", "Perry"]
WEATHER_MASTER = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
DOOF_MASTER = ["None", "Shrink-inator", "Magnify-inator", "Erase-inator", "Attractor-inator", "Deja Vu-inator"]
PERRY_MASTER = ["Unknown", "At home", "On a mission"]
CANDACE_MASTER = ["Chill", "Suspicious", "Determined to Bust"]

@st.cache_data
def load_seed_dataset():
    data = [
        # minimal playful seed data (feel free to expand via UI)
        {"weather":"Sunny","time":8,"crew":"Isabella;Perry","materials":"Wood;Electronics;Gadgets","backyard_only":True,"perry":"On a mission","doof":"None","candace":"Chill","project":"Backyard Roller Coaster 2.0"},
        {"weather":"Windy","time":5,"crew":"Isabella","materials":"Electronics","backyard_only":False,"perry":"Unknown","doof":"None","candace":"Suspicious","project":"Skywriting Drone Swarm"},
        {"weather":"Cloudy","time":3,"crew":"Baljeet;Perry","materials":"Electronics;Gadgets","backyard_only":True,"perry":"On a mission","doof":"Shrink-inator","candace":"Chill","project":"Doof-Inator Disruptor Shield"},
        {"weather":"Sunny","time":10,"crew":"Buford","materials":"Metal;Electronics","backyard_only":False,"perry":"At home","doof":"Magnify-inator","candace":"Determined to Bust","project":"Suburban Space Elevator"},
        {"weather":"Rainy","time":4,"crew":"Isabella;Candace","materials":"Rubber Bands;Electronics","backyard_only":True,"perry":"Unknown","doof":"None","candace":"Suspicious","project":"Ocean-Wave Surf Simulator"},
        {"weather":"Snowy","time":2,"crew":"Perry","materials":"Paint;Wood","backyard_only":True,"perry":"On a mission","doof":"Erase-inator","candace":"Chill","project":"DIY Snow-Globe City"},
        {"weather":"Sunny","time":7,"crew":"Baljeet;Buford","materials":"Jet Fuel (uh-oh);Metal","backyard_only":False,"perry":"At home","doof":"Attractor-inator","candace":"Chill","project":"Rainbow-Powered Jetpacks"},
        {"weather":"Cloudy","time":9,"crew":"Isabella;Baljeet","materials":"Ice Cream;Wood;Gadgets","backyard_only":True,"perry":"Unknown","doof":"None","candace":"Chill","project":"Giant Rube Goldberg Ice-Cream Machine"},
        {"weather":"Sunny","time":11,"crew":"Perry;Buford","materials":"Metal","backyard_only":False,"perry":"On a mission","doof":"None","candace":"Suspicious","project":"Underground Hyper-Tunnel"},
        {"weather":"Windy","time":6,"crew":"Isabella;Baljeet","materials":"Electronics;Gadgets","backyard_only":False,"perry":"At home","doof":"Deja Vu-inator","candace":"Chill","project":"Time-Traveling Sandwich Maker"},
    ]
    return pd.DataFrame(data)

DATA_PATH = "episodes.csv"

def get_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        # Normalize any multi-select columns
        for col in ["crew","materials"]:
            df[col] = df[col].fillna("")
            df[col] = df[col].astype(str).str.replace(",", ";")
        return df
    except Exception:
        df = load_seed_dataset()
        df.to_csv(DATA_PATH, index=False)
        return df

def save_dataset(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

df = get_dataset()

# ----------------------
# Sidebar: data tools
# ----------------------
with st.sidebar:
    st.markdown("### 📚 Dataset Tools")
    st.caption("Use or expand a tiny training dataset for smarter predictions.")
    st.dataframe(df.tail(5), use_container_width=True)
    uploaded = st.file_uploader("Upload episodes.csv to replace dataset", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        save_dataset(df)
        st.success("Dataset replaced from uploaded file.")

    if st.button("⬇️ Download current dataset as CSV"):
        st.download_button("Download episodes.csv", data=df.to_csv(index=False), file_name="episodes.csv", mime="text/csv")

# ----------------------
# Input form (same fields as earlier)
# ----------------------
with st.form("inputs"):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        the_date = st.date_input("Day in Danville", value=date.today(), help="Set the day for this 'episode'.")
        weather = st.selectbox("Weather", WEATHER_MASTER)
        time_available = st.slider("Free time (hours)", 1, 12, 6)
    with col2:
        crew = st.multiselect("Who's around?", CREW_MASTER, default=["Isabella", "Perry"])
        materials = st.multiselect("Materials on hand", MATERIALS_MASTER, default=["Wood", "Electronics", "Gadgets"])
        backyard_only = st.toggle("Backyard build only", value=True)
    with col3:
        perry_status = st.select_slider("Where's Perry?", options=PERRY_MASTER, value="On a mission")
        doof_inator = st.selectbox("Doof's *-inator* of the day", DOOF_MASTER)
        candace_mood = st.selectbox("Candace's mood", CANDACE_MASTER)

    colx, coly = st.columns([1,1])
    with colx:
        submitted = st.form_submit_button("🎉 Predict with AI", use_container_width=True)
    with coly:
        add_row = st.form_submit_button("➕ Add this as a training episode", use_container_width=True)

# ----------------------
# Feature engineering
# ----------------------
def encode_rows(rows: pd.DataFrame):
    # Multi-hot for crew + materials
    crew_split = rows['crew'].fillna('').apply(lambda s: [x.strip() for x in str(s).split(';') if x.strip()])
    mat_split  = rows['materials'].fillna('').apply(lambda s: [x.strip() for x in str(s).split(';') if x.strip()])

    mlb_crew = MultiLabelBinarizer(classes=CREW_MASTER)
    mlb_mat  = MultiLabelBinarizer(classes=MATERIALS_MASTER)

    X_crew = mlb_crew.fit_transform(crew_split)
    X_mat  = mlb_mat.fit_transform(mat_split)

    # One-hot for categorical singles
    oh = OneHotEncoder(categories=[WEATHER_MASTER, [True, False], PERRY_MASTER, DOOF_MASTER, CANDACE_MASTER], sparse_output=False, handle_unknown='ignore')
    cat = rows[['weather','backyard_only','perry','doof','candace']].copy()
    X_cat = oh.fit_transform(cat)

    # Numeric time, scaled to 0-1
    time_scaled = (rows[['time']].astype(float) / 12.0).to_numpy()

    # Final feature matrix
    X = np.hstack([time_scaled, X_crew, X_mat, X_cat])
    return X

# ----------------------
# Add row to dataset
# ----------------------
if add_row:
    new_row = {
        "weather": weather,
        "time": time_available,
        "crew": ";".join(crew),
        "materials": ";".join(materials),
        "backyard_only": backyard_only,
        "perry": perry_status,
        "doof": doof_inator,
        "candace": candace_mood,
        "project": ""  # you can label later after you see what they actually built
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_dataset(df)
    st.success("Episode added to dataset. You can label the 'project' later in episodes.csv.")

# ----------------------
# Deterministic RNG (ties / aesthetics)
# ----------------------
def deterministic_rng(seed_text: str) -> random.Random:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed_text)  # seed with text for reproducibility

# ----------------------
# AI Prediction via k-NN on one-hot features
# ----------------------
def ai_predict():
    if df.empty:
        st.warning("Dataset is empty. Add some training rows first.")
        return None, []

    # Build X for train
    train = df.copy()
    X_train = encode_rows(train.rename(columns={"backyard_only":"backyard_only"}))

    # Query vector
    query = pd.DataFrame([{
        "weather": weather,
        "time": time_available,
        "crew": ";".join(crew),
        "materials": ";".join(materials),
        "backyard_only": backyard_only,
        "perry": perry_status,
        "doof": doof_inator,
        "candace": candace_mood,
        "project": ""
    }])
    X_query = encode_rows(query)

    # Fit k-NN (cosine distance)
    k = min(5, len(train))
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_train)
    dist, idx = nn.kneighbors(X_query)

    # Aggregate neighbor votes
    neighbors = train.iloc[idx[0]].copy()
    neighbors['weight'] = 1 - dist[0]  # similarity

    # If labeled projects exist among neighbors, do weighted vote
    labeled = neighbors[neighbors['project'].astype(str).str.len() > 0]
    if not labeled.empty:
        agg = labeled.groupby('project')['weight'].sum().sort_values(ascending=False)
        top_project = agg.index[0]
        alt = agg.index[1:4].tolist()
        return top_project, alt

    # Otherwise fall back to rule-based pick from DEFAULT_PROJECTS
    rng = deterministic_rng(str(query.to_dict()))
    candidates = DEFAULT_PROJECTS.copy()
    rng.shuffle(candidates)
    return candidates[0]['name'], [c['name'] for c in candidates[1:4]]

# ----------------------
# Run prediction
# ----------------------
if submitted:
    top, alts = ai_predict()
    if top is None:
        st.stop()

    # Output cards
    st.markdown('<div class="cartoon-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Prediction")
    st.markdown(f"**{top}**")
    if alts:
        st.caption("Runner-ups: " + " • ".join(alts))
    st.markdown('</div>', unsafe_allow_html=True)

    # Fun blueprint
    blueprint_lines = [
        f"- Weather: {weather}",
        f"- Time: {time_available}h",
        f"- Crew: {', '.join(crew) if crew else 'Just the brothers'}",
        f"- Materials: {', '.join(materials) if materials else 'Garage stash'}",
        f"- Backyard only: {'Yes' if backyard_only else 'No'}",
        f"- Perry: {perry_status}",
        f"- Doof: {doof_inator}",
        f"- Candace: {candace_mood}",
    ]
    st.markdown('<div class="cartoon-card blueprint">', unsafe_allow_html=True)
    st.subheader("📐 Blueprint Notes")
    st.text("\n".join(blueprint_lines))
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Labeling helper
# ----------------------
st.markdown('<div class="cartoon-card">', unsafe_allow_html=True)
st.subheader("✍️ Label an episode (teaches the AI)")
if not df.empty:
    row_idx = st.number_input("Row to label (0-based index)", min_value=0, max_value=len(df)-1, value=len(df)-1, step=1)
    choice = st.selectbox("Project name", PROJECT_NAMES)
    if st.button("Save label"):
        df.loc[row_idx, 'project'] = choice
        save_dataset(df)
        st.success(f"Labeled row {row_idx} as '{choice}'. The model will use it on the next run.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="cartoon-card footer-note">Tip: grow your dataset by adding rows daily; labeling improves predictions fast.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
