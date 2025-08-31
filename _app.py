import streamlit as st
import random
import hashlib
from datetime import date

    # ----------------------
    # Page config
    # ----------------------
st.set_page_config(
    page_title="Phineas & Ferb: Project Predictor",
    page_icon="🛠️",
    layout="wide",
)

    # ----------------------
    # Cartoon-styled CSS
    # ----------------------
CARTOON_CSS = """
    <link href="https://fonts.googleapis.com/css2?family=Comic+Neue:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"]  {
        font-family: 'Comic Neue', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      }
      .cartoon-bg {
        background:
          radial-gradient(circle at 15% 20%, #fff7a1 0 20%, transparent 21% 100%),
          radial-gradient(circle at 85% 10%, #ffd6e7 0 18%, transparent 19% 100%),
          radial-gradient(circle at 75% 85%, #c4f2ff 0 22%, transparent 23% 100%),
          radial-gradient(circle at 10% 80%, #c3ffa8 0 18%, transparent 19% 100%),
          linear-gradient(180deg, #f2f7ff, #e9f5ff);
        min-height: 100vh;
        padding-bottom: 2rem;
      }
      .cartoon-card {
        background: #fff;
        border: 4px solid #111;
        border-radius: 18px;
        box-shadow: 6px 6px 0 #111;
        padding: 1.2rem 1.2rem 0.6rem 1.2rem;
        margin-bottom: 1rem;
      }
      .cartoon-title {
        display: inline-block;
        background: #ffe66d;
        border: 4px solid #111;
        border-radius: 16px;
        padding: 8px 14px;
        box-shadow: 6px 6px 0 #111;
        font-weight: 800;
        letter-spacing: 0.5px;
      }
      .badge {
        display: inline-block;
        background: #9bf6ff;
        border: 3px solid #111;
        border-radius: 999px;
        padding: 4px 10px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 700;
      }
      .big-btn {
        display: inline-block;
        background: linear-gradient(0deg, #ffb703, #ffd166);
        border: 4px solid #111;
        border-radius: 16px;
        padding: 12px 18px;
        box-shadow: 6px 6px 0 #111;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        cursor: pointer;
      }
      .big-btn:active {
        transform: translate(1px, 1px);
        box-shadow: 4px 4px 0 #111;
      }
      .blueprint {
        background: repeating-linear-gradient(
          0deg,
          #0a2540 0px,
          #0a2540 2px,
          #0d335e 2px,
          #0d335e 34px
        );
        color: #c7e0ff;
        border: 4px solid #001d3d;
        border-radius: 14px;
        padding: 16px;
        font-family: "Courier New", monospace;
        box-shadow: 6px 6px 0 #001d3d;
      }
      .footer-note {
        opacity: 0.8;
        font-size: 0.9rem;
      }
    </style>
    """

st.markdown('<div class="cartoon-bg">', unsafe_allow_html=True)
st.markdown(CARTOON_CSS, unsafe_allow_html=True)

    # ----------------------
    # Title
    # ----------------------
st.markdown('<h1 class="cartoon-title">Phineas & Ferb: Project Predictor 🛠️🚀</h1>', unsafe_allow_html=True)
st.write("")

    # ----------------------
    # Sidebar (cartoon badges + tips)
    # ----------------------
with st.sidebar:
    st.markdown("### 🎬 Cast & Variables")
    st.markdown('<span class="badge">Candace</span><span class="badge">Isabella</span><span class="badge">Baljeet</span><span class="badge">Buford</span><span class="badge">Perry</span><span class="badge">Doof</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### 🧪 Cartoon Lab Tips")
    st.markdown("- Bigger ideas happen on **sunny** days.- **Perry** nearby ⇒ more secret-agent chaos.- **Doof** with an *-inator* ⇒ expect counter-inventions.- More **time & materials** ⇒ bigger builds.")

    # ----------------------
    # Input form
    # ----------------------
with st.container():
    with st.form("inputs"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            the_date = st.date_input("Day in Danville", value=date.today(), help="Set the day for this 'episode'.")
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"])
            time_available = st.slider("Free time (hours)", 1, 12, 6)
        with col2:
            crew = st.multiselect(
                "Who's around?",
                ["Candace", "Isabella", "Baljeet", "Buford", "Perry"],
                default=["Isabella", "Perry"]
            )
            materials = st.multiselect(
                "Materials on hand",
                ["Wood", "Metal", "Electronics", "Jet Fuel (uh-oh)", "Gadgets", "Paint", "Ice Cream", "Rubber Bands"],
                default=["Wood", "Electronics", "Gadgets"]
            )
            backyard_only = st.toggle("Backyard build only", value=True)
        with col3:
            perry_status = st.select_slider("Where's Perry?", options=["Unknown", "At home", "On a mission"], value="On a mission")
            doof_inator = st.selectbox("Doof's *-inator* of the day", ["None", "Shrink-inator", "Magnify-inator", "Erase-inator", "Attractor-inator", "Deja Vu-inator"])
            candace_mood = st.selectbox("Candace's mood", ["Chill", "Suspicious", "Determined to Bust"])

        submitted = st.form_submit_button("🎉 Predict Today's Project!", use_container_width=True)

    # ----------------------
    # Prediction logic
    # ----------------------
def deterministic_rng(seed_text: str) -> random.Random:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)

PROJECTS = [
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
    {"name": "Backyard Biodome", "scale": 5, "tags": ["Wood", "Metal", "Paint"], "needs_space": True},
    {"name": "Giant Slingshot Launcher", "scale": 4, "tags": ["Rubber Bands", "Wood"], "needs_space": True},
    {"name": "Holographic Concert Stage", "scale": 3, "tags": ["Electronics", "Gadgets"], "needs_space": False},
    {"name": "Automated Homework Robot", "scale": 2, "tags": ["Electronics", "Gadgets"], "needs_space": False},
    {"name": "Personal Weather Control System", "scale": 6, "tags": ["Electronics", "Gadgets", "Metal"], "needs_space": False}, 
]

def score_project(p, rng: random.Random) -> float:
    score = 0.0
        # Materials synergy
    overlap = len(set(p["tags"]).intersection(set(materials)))
    score += overlap * 2.5

        # Weather synergy
    if weather == "Sunny":
        score += 2 if p["scale"] >= 4 else 1
    elif weather in {"Rainy", "Snowy"} and p["needs_space"]:
        score -= 2
    elif weather == "Windy" and "Skywriting" in p["name"]:
        score += 2

        # Time
    if time_available >= 8 and p["scale"] >= 5:
        score += 2.5
    elif time_available <= 3 and p["scale"] >= 5:
        score -= 2

        # Backyard constraint
    if backyard_only and p["needs_space"]:
        score -= 1.5

        # Cast effects
    if "Isabella" in crew and "Drone" in p["name"]:
        score += 1.5
    if "Baljeet" in crew and "Time-Traveling" in p["name"]:
        score += 1.2
    if "Buford" in crew and "Roller Coaster" in p["name"]:
        score += 1.2

        # Doof & Perry chaos
    if doof_inator != "None" and "Disruptor" in p["name"]:
        score += 2.2
    if perry_status == "On a mission":
        score += rng.uniform(-0.8, 0.8)  # unpredictable!
    if candace_mood == "Determined to Bust":
        score += rng.uniform(-1.5, 0.3)

        # Small random nudge for variety
    score += rng.uniform(-0.6, 0.6)
    return score

if submitted:
    seed_text = f"{the_date}|{weather}|{time_available}|{sorted(crew)}|{sorted(materials)}|{backyard_only}|{perry_status}|{doof_inator}|{candace_mood}"
    rng = deterministic_rng(seed_text)
    scored = sorted(
        [(p, score_project(p, rng)) for p in PROJECTS],
        key=lambda x: x[1],
        reverse=True
    )
    top_project, top_score = scored[0]

        # Create a playful episode title
    title_bits = [
        "The One With The",
        "A Totally Normal Day Of",
        "Operation",
        "Project",
        "Summer Saga:",
        "The (Not So) Great",
        "Mission:",
        "The (Un)Incredible",
        "The (Mis)Adventures of",
        "The (Un)Believable",
        "The (Un)Real",
        "The (Un)Expected",
        "The (Un)Imaginable",
    ]
    ep_title = f"{rng.choice(title_bits)} {top_project['name']}"

        # "Blueprint" text block
    blueprint_lines = [
        f"- Core idea: {top_project['name']}",
        f"- Weather boost: {weather}",
        f"- Crew: {', '.join(crew) if crew else 'Just the brothers'}",
        f"- Materials: {', '.join(materials) if materials else 'Whatever is in the garage'}",
        f"- Backyard only: {'Yes' if backyard_only else 'No'}",
        f"- Doof's device: {doof_inator}",
        f"- Where's Perry: {perry_status}",
        f"- Candace factor: {candace_mood}",
        f"- Build scale: {top_project['scale']}/6",
    ]

    colA, colB = st.columns([1,1])
    with colA:
        st.markdown('<div class="cartoon-card">', unsafe_allow_html=True)
        st.subheader("🎯 Today’s Predicted Project")
        st.markdown(f"**{top_project['name']}**")
        st.caption(f"(Score: {top_score:.2f} — higher is more likely)")
        st.progress(min(max((top_score + 3) / 8, 0), 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cartoon-card">', unsafe_allow_html=True)
        st.subheader("📺 Episode Title")
        st.markdown(f"***{ep_title}***")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="cartoon-card blueprint">', unsafe_allow_html=True)
        st.subheader("📐 Blueprint Notes")
        st.text("\n".join(blueprint_lines))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cartoon-card">', unsafe_allow_html=True)
    st.subheader("🎲 Runners-Up (Honorable Mentions)")
    for p, s in scored[1:4]:
        st.markdown(f"- **{p['name']}** — score {s:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cartoon-card footer-note">', unsafe_allow_html=True)
    st.write("Tip: Same inputs ⇒ same prediction (deterministic). Try tweaking the crew, weather, or Doof’s *-inator*!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
