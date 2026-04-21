"""
SHAP Diabetes AI — Flask Deployment App
Real-time predictions with SHAP explanations
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import shap
import json
import os

app = Flask(__name__)

# Load artifacts
model   = joblib.load('models/xgb_model.pkl')
scaler  = joblib.load('models/scaler.pkl')
FEATURES = joblib.load('models/features.pkl')
explainer = shap.TreeExplainer(model)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHAP Diabetes AI — Prediction</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #21262d;
    --accent: #00d4aa; --danger: #ff6b6b; --warn: #ffd166;
    --text: #e6edf3; --muted: #8b949e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         min-height: 100vh; padding: 30px 20px; }
  h1 { text-align: center; font-size: 2rem; background: linear-gradient(135deg, var(--accent), #00aaff);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px; }
  .sub { text-align: center; color: var(--muted); margin-bottom: 30px; font-size: .95rem; }
  .container { max-width: 860px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }
  label { display: block; font-size: .82rem; color: var(--muted); margin-bottom: 5px; font-weight: 600; letter-spacing: .04em; }
  input[type=number] { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: 10px 12px; font-size: .95rem; outline: none; transition: border .2s; }
  input:focus { border-color: var(--accent); }
  .hint { font-size: .72rem; color: var(--muted); margin-top: 3px; }
  button { width: 100%; background: linear-gradient(135deg, var(--accent), #00aaff); border: none;
    border-radius: 10px; color: #0d1117; font-weight: 700; font-size: 1.05rem; padding: 14px;
    cursor: pointer; transition: opacity .2s, transform .1s; margin-top: 8px; }
  button:hover { opacity: .9; transform: translateY(-1px); }
  #result { display: none; }
  .risk-badge { font-size: 1.6rem; font-weight: 800; text-align: center; padding: 18px;
    border-radius: 10px; margin-bottom: 16px; }
  .risk-low  { background: rgba(0,212,170,.15); color: var(--accent); border: 1px solid var(--accent); }
  .risk-high { background: rgba(255,107,107,.15); color: var(--danger); border: 1px solid var(--danger); }
  .risk-med  { background: rgba(255,209,102,.15); color: var(--warn);   border: 1px solid var(--warn); }
  .shap-row { display: flex; align-items: center; margin: 7px 0; gap: 10px; }
  .shap-label { width: 200px; font-size: .85rem; text-align: right; flex-shrink: 0; }
  .shap-bar-wrap { flex: 1; height: 22px; background: var(--border); border-radius: 4px; overflow: hidden;
    position: relative; }
  .shap-bar { height: 100%; border-radius: 4px; transition: width .5s; }
  .shap-val { font-size: .8rem; color: var(--muted); width: 60px; text-align: left; flex-shrink: 0; }
  .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; text-align: center; margin-top: 8px; }
  .metric-val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
  .metric-lab { font-size: .75rem; color: var(--muted); margin-top: 2px; }
  footer { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 40px; }
</style>
</head>
<body>
<div class="container">
  <h1>🩺 SHAP Diabetes AI</h1>
  <p class="sub">Enter patient data to get an explainable diabetes risk prediction</p>

  <div class="card">
    <div class="grid">
      <div>
        <label>Pregnancies</label>
        <input type="number" id="Pregnancies" value="2" min="0" max="20">
        <div class="hint">Number of pregnancies</div>
      </div>
      <div>
        <label>Glucose (mg/dL)</label>
        <input type="number" id="Glucose" value="120" min="0" max="300">
        <div class="hint">Plasma glucose (2hr OGTT)</div>
      </div>
      <div>
        <label>Blood Pressure (mmHg)</label>
        <input type="number" id="BloodPressure" value="72" min="0" max="200">
        <div class="hint">Diastolic blood pressure</div>
      </div>
      <div>
        <label>Skin Thickness (mm)</label>
        <input type="number" id="SkinThickness" value="23" min="0" max="100">
        <div class="hint">Triceps skin fold thickness</div>
      </div>
      <div>
        <label>Insulin (mu U/ml)</label>
        <input type="number" id="Insulin" value="80" min="0" max="900">
        <div class="hint">2-Hour serum insulin</div>
      </div>
      <div>
        <label>BMI (kg/m²)</label>
        <input type="number" id="BMI" value="32.5" step="0.1" min="0" max="80">
        <div class="hint">Body Mass Index</div>
      </div>
      <div>
        <label>Diabetes Pedigree Fn</label>
        <input type="number" id="DiabetesPedigreeFunction" value="0.35" step="0.001" min="0" max="3">
        <div class="hint">Family history score</div>
      </div>
      <div>
        <label>Age (years)</label>
        <input type="number" id="Age" value="35" min="1" max="120">
        <div class="hint">Patient age</div>
      </div>
    </div>
    <button onclick="predict()">🔍 Analyze Risk with SHAP</button>
  </div>

  <div class="card" id="result">
    <div id="riskBadge" class="risk-badge"></div>
    <div class="metrics">
      <div><div class="metric-val" id="probVal">—</div><div class="metric-lab">Diabetes Risk %</div></div>
      <div><div class="metric-val" id="baseVal">—</div><div class="metric-lab">Baseline Risk %</div></div>
      <div><div class="metric-val" id="decisionVal">—</div><div class="metric-lab">Decision</div></div>
    </div>
    <hr style="border-color:var(--border);margin:20px 0">
    <h3 style="margin-bottom:12px;color:var(--warn)">🔍 SHAP Feature Contributions</h3>
    <p style="font-size:.82rem;color:var(--muted);margin-bottom:14px">
      Green bars push risk ↓ (protective) · Red bars push risk ↑ (risk-increasing)
    </p>
    <div id="shapChart"></div>
  </div>
</div>

<footer>Built with XGBoost + SHAP · Pima Indians Diabetes Dataset · AUC 0.947</footer>

<script>
async function predict() {
  const fields = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                  'Insulin','BMI','DiabetesPedigreeFunction','Age'];
  const data = {};
  for (const f of fields) data[f] = parseFloat(document.getElementById(f).value) || 0;

  const resp = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  const res = await resp.json();
  if (res.error) { alert(res.error); return; }

  document.getElementById('result').style.display = 'block';
  const prob = (res.probability * 100).toFixed(1);
  const base = (res.base_value * 100).toFixed(1);
  document.getElementById('probVal').textContent = prob + '%';
  document.getElementById('baseVal').textContent = base + '%';
  document.getElementById('decisionVal').textContent = res.prediction === 1 ? '⚠️ Diabetic' : '✅ Non-Diabetic';

  const badge = document.getElementById('riskBadge');
  if (res.probability > 0.65) {
    badge.className = 'risk-badge risk-high';
    badge.textContent = `⚠️ HIGH RISK — ${prob}% Probability of Diabetes`;
  } else if (res.probability > 0.35) {
    badge.className = 'risk-badge risk-med';
    badge.textContent = `⚡ MODERATE RISK — ${prob}% Probability of Diabetes`;
  } else {
    badge.className = 'risk-badge risk-low';
    badge.textContent = `✅ LOW RISK — ${prob}% Probability of Diabetes`;
  }

  // SHAP chart
  const shap = res.shap_values;
  const sorted = Object.entries(shap).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(s => Math.abs(s[1])));
  let html = '';
  for (const [feat, val] of sorted) {
    const pct = (Math.abs(val) / maxAbs * 100).toFixed(1);
    const color = val > 0 ? '#ff6b6b' : '#00d4aa';
    const dir = val > 0 ? '↑' : '↓';
    html += `
      <div class="shap-row">
        <div class="shap-label">${feat}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="shap-val">${dir} ${Math.abs(val).toFixed(3)}</div>
      </div>`;
  }
  document.getElementById('shapChart').innerHTML = html;
  document.getElementById('result').scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        base_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']

        row = {col: float(data.get(col, 0)) for col in base_cols}

        # Feature engineering (must match training)
        row['GlucoseBMI']    = row['Glucose'] * row['BMI'] / 100
        row['AgePregRatio']  = row['Age'] / (row['Pregnancies'] + 1)
        row['InsulinResist'] = row['Insulin'] / (row['Glucose'] + 1)
        row['MetabolicRisk'] = (row['Glucose'] / 100) * (row['BMI'] / 25)

        X = pd.DataFrame([row])[FEATURES]
        X_s = scaler.transform(X)

        prob = model.predict_proba(X_s)[0][1]
        pred = int(prob >= 0.5)

        sv = explainer.shap_values(X_s)[0]
        shap_dict = {FEATURES[i]: round(float(sv[i]), 4) for i in range(len(FEATURES))}

        return jsonify({
            'prediction': pred,
            'probability': round(float(prob), 4),
            'base_value': round(float(explainer.expected_value), 4),
            'shap_values': shap_dict,
            'risk_level': 'HIGH' if prob > 0.65 else 'MODERATE' if prob > 0.35 else 'LOW'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost', 'auc': 0.9474})

if __name__ == '__main__':
    print("\n🩺 SHAP Diabetes AI — Starting server on http://localhost:5000\n")
    app.run(debug=True, port=5000)
