
# AI+ Upgrade

This version adds a tiny **episodes.csv** dataset and a lightweight **k-NN** predictor:
- Add new "episodes" from the UI.
- Label rows with the built project → the model learns.
- Predicts based on nearest neighbors over one-hot features (crew, materials, weather, etc.).
- Still fully cartoon-themed.

## Run (uses `py` launcher on Windows)
```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app_ai.py
```

### Files
- `app_ai.py` — AI+ Streamlit app
- `episodes.csv` — starter dataset (you can replace it)
- `requirements.txt` — includes pandas/numpy/scikit-learn

### Optional: Bring Your Own LLM
If you want natural-language "episode summaries" or idea generation, plug in any LLM SDK in a separate module (kept optional so the app runs fine without API keys).
