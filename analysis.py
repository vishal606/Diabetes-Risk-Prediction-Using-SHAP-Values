"""
SHAP Diabetes Analysis — Full Pipeline
EDA → Feature Engineering → Model Training → SHAP Explainability → Plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── Style ───────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
ACCENT    = '#00d4aa'
ACCENT2   = '#ff6b6b'
ACCENT3   = '#ffd166'
TEXT      = '#e6edf3'
CARD      = '#161b22'
GRID      = '#21262d'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': CARD,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID,
    'grid.alpha': 0.5, 'font.family': 'DejaVu Sans',
})

# ─── 1. Load & Clean ─────────────────────────────────────────────────────────
print("=" * 60)
print("  SHAP DIABETES AI — Full Analysis Pipeline")
print("=" * 60)

df = pd.read_csv('data/diabetes.csv')
print(f"\n[DATA] Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# Replace 0s with NaN for biological impossibilities
bio_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[bio_cols] = df[bio_cols].replace(0, np.nan)

# Impute with median (stratified)
for col in bio_cols:
    df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

print(f"[CLEAN] Missing values handled via stratified median imputation")

# ─── 2. Feature Engineering ──────────────────────────────────────────────────
df['GlucoseBMI']     = df['Glucose'] * df['BMI'] / 100
df['AgePregRatio']   = df['Age'] / (df['Pregnancies'] + 1)
df['InsulinResist']  = df['Insulin'] / (df['Glucose'] + 1)
df['MetabolicRisk']  = (df['Glucose'] / 100) * (df['BMI'] / 25)

print(f"[FEAT] 4 engineered features added. Total features: {df.shape[1]-1}")

FEATURES = [c for c in df.columns if c != 'Outcome']
X = df[FEATURES]
y = df['Outcome']

# ─── 3. EDA Plots ────────────────────────────────────────────────────────────
print("\n[EDA] Generating plots...")

# 3a. Distribution of all features
fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor=DARK_BG)
fig.suptitle('Feature Distributions by Diabetes Outcome', fontsize=16,
             color=TEXT, fontweight='bold', y=1.01)
colors = {0: ACCENT, 1: ACCENT2}
labels = {0: 'No Diabetes', 1: 'Diabetes'}

for idx, col in enumerate(FEATURES):
    ax = axes[idx // 4][idx % 4]
    for outcome in [0, 1]:
        subset = df[df['Outcome'] == outcome][col].dropna()
        ax.hist(subset, bins=25, alpha=0.65, color=colors[outcome],
                label=labels[outcome], density=True, edgecolor='none')
    ax.set_title(col, fontsize=10, color=ACCENT3, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/01_feature_distributions.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Feature distributions saved")

# 3b. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9), facecolor=DARK_BG)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f',
            linewidths=0.5, linecolor=DARK_BG, ax=ax,
            annot_kws={'size': 8}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=14, color=TEXT,
             fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Correlation heatmap saved")

# 3c. Class balance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
counts = y.value_counts()
bars = ax1.bar(['No Diabetes', 'Diabetes'], counts.values,
               color=[ACCENT, ACCENT2], width=0.5, edgecolor='none', alpha=0.9)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}\n({val/len(y)*100:.1f}%)', ha='center', va='bottom',
             color=TEXT, fontweight='bold')
ax1.set_title('Class Distribution', fontsize=13, color=TEXT, fontweight='bold')
ax1.set_ylabel('Count', color=TEXT)
ax1.grid(axis='y', alpha=0.3)

wedges, texts, autotexts = ax2.pie(counts.values, labels=['No Diabetes', 'Diabetes'],
    colors=[ACCENT, ACCENT2], autopct='%1.1f%%', startangle=90,
    textprops={'color': TEXT}, pctdistance=0.75,
    wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
ax2.set_title('Outcome Ratio', fontsize=13, color=TEXT, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/03_class_balance.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Class balance saved")

# 3d. Box plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=DARK_BG)
for idx, col in enumerate(FEATURES[:8]):
    ax = axes[idx // 4][idx % 4]
    data_no = df[df['Outcome'] == 0][col].dropna()
    data_yes = df[df['Outcome'] == 1][col].dropna()
    bp = ax.boxplot([data_no, data_yes], patch_artist=True,
                    medianprops={'color': ACCENT3, 'linewidth': 2},
                    whiskerprops={'color': TEXT}, capprops={'color': TEXT},
                    boxprops={'edgecolor': TEXT})
    bp['boxes'][0].set_facecolor(ACCENT + '80')
    bp['boxes'][1].set_facecolor(ACCENT2 + '80')
    ax.set_xticklabels(['No DM', 'DM'])
    ax.set_title(col, fontsize=9, color=ACCENT3, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Feature Box Plots by Diabetes Status', fontsize=14,
             color=TEXT, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_boxplots.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Box plots saved")

# ─── 4. Model Training ───────────────────────────────────────────────────────
print("\n[MODEL] Training models...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# XGBoost (primary for SHAP)
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
    eval_metric='logloss', random_state=42
)
xgb_model.fit(X_train_s, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train_s, y_train)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv = cross_val_score(xgb_model, X_train_s, y_train, cv=skf, scoring='roc_auc')
rf_cv  = cross_val_score(rf_model,  X_train_s, y_train, cv=skf, scoring='roc_auc')

print(f"  XGBoost CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"  RandomForest CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# Evaluate on test set
y_pred_xgb  = xgb_model.predict(X_test_s)
y_prob_xgb  = xgb_model.predict_proba(X_test_s)[:, 1]
y_pred_rf   = rf_model.predict(X_test_s)
y_prob_rf   = rf_model.predict_proba(X_test_s)[:, 1]

print(f"\n  XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"  XGBoost Test AUC:      {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(f"\n  RF Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  RF Test AUC:      {roc_auc_score(y_test, y_prob_rf):.4f}")

# ─── 5. ROC Curve ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
for model_name, y_prob, color in [
        ('XGBoost', y_prob_xgb, ACCENT),
        ('Random Forest', y_prob_rf, ACCENT2)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], '--', color=GRID, lw=1.5, label='Random Classifier')
ax.fill_between(*roc_curve(y_test, y_prob_xgb)[:2], alpha=0.1, color=ACCENT)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Diabetes Prediction Models', fontsize=14,
             color=TEXT, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/05_roc_curve.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("\n  ✓ ROC curve saved")

# ─── 6. Confusion Matrix ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
for ax, y_pred, title in [(ax1, y_pred_xgb, 'XGBoost'), (ax2, y_pred_rf, 'Random Forest')]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                linewidths=2, linecolor=DARK_BG,
                xticklabels=['No DM', 'DM'], yticklabels=['No DM', 'DM'])
    ax.set_title(f'Confusion Matrix — {title}', color=TEXT, fontweight='bold')
    ax.set_xlabel('Predicted', color=TEXT)
    ax.set_ylabel('Actual', color=TEXT)
plt.tight_layout()
plt.savefig('plots/06_confusion_matrix.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ Confusion matrix saved")

# ─── 7. SHAP Analysis ────────────────────────────────────────────────────────
print("\n[SHAP] Computing SHAP values (this may take ~30s)...")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_s)
X_test_df = pd.DataFrame(X_test_s, columns=FEATURES)

# 7a. SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 8), facecolor=DARK_BG)
shap.summary_plot(shap_values, X_test_df, plot_type='dot',
                  show=False, color_bar=True, max_display=12)
plt.title('SHAP Summary — Feature Impact on Diabetes Prediction',
          fontsize=14, color=TEXT, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/07_shap_summary_beeswarm.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP beeswarm summary saved")

# 7b. SHAP Bar Plot (Mean |SHAP|)
plt.figure(figsize=(10, 7), facecolor=DARK_BG)
shap.summary_plot(shap_values, X_test_df, plot_type='bar',
                  show=False, max_display=12)
plt.title('SHAP Feature Importance (Mean |SHAP Value|)',
          fontsize=14, color=TEXT, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/08_shap_bar_importance.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP bar importance saved")

# 7c. SHAP Waterfall — single high-risk patient
high_risk_idx = np.argmax(y_prob_xgb)
plt.figure(figsize=(10, 7), facecolor=DARK_BG)
shap_exp = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=X_test_df.iloc[high_risk_idx].values,
    feature_names=FEATURES
)
shap.waterfall_plot(shap_exp, show=False, max_display=12)
plt.title(f'SHAP Waterfall — High-Risk Patient (Pred: {y_prob_xgb[high_risk_idx]:.1%})',
          fontsize=13, color=TEXT, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('plots/09_shap_waterfall_highrisk.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP waterfall (high-risk) saved")

# 7d. SHAP Waterfall — low-risk patient
low_risk_idx = np.argmin(y_prob_xgb)
plt.figure(figsize=(10, 7), facecolor=DARK_BG)
shap_exp_low = shap.Explanation(
    values=shap_values[low_risk_idx],
    base_values=explainer.expected_value,
    data=X_test_df.iloc[low_risk_idx].values,
    feature_names=FEATURES
)
shap.waterfall_plot(shap_exp_low, show=False, max_display=12)
plt.title(f'SHAP Waterfall — Low-Risk Patient (Pred: {y_prob_xgb[low_risk_idx]:.1%})',
          fontsize=13, color=TEXT, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('plots/10_shap_waterfall_lowrisk.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP waterfall (low-risk) saved")

# 7e. SHAP Dependence Plots — top 2 features
mean_shap = np.abs(shap_values).mean(0)
top2 = np.argsort(mean_shap)[-2:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
for i, feat_idx in enumerate(top2):
    feat_name = FEATURES[feat_idx]
    sc = axes[i].scatter(X_test_df.iloc[:, feat_idx], shap_values[:, feat_idx],
                         c=shap_values[:, feat_idx], cmap='RdYlGn',
                         alpha=0.7, s=20, edgecolors='none')
    axes[i].axhline(0, color=GRID, lw=1.5, linestyle='--')
    axes[i].set_xlabel(feat_name, fontsize=11)
    axes[i].set_ylabel('SHAP Value', fontsize=11)
    axes[i].set_title(f'SHAP Dependence — {feat_name}',
                      fontsize=12, color=ACCENT3, fontweight='bold')
    axes[i].grid(True, alpha=0.25)
    plt.colorbar(sc, ax=axes[i], label='SHAP value')
plt.suptitle('SHAP Dependence Plots — Top Features', fontsize=14,
             color=TEXT, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/11_shap_dependence.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP dependence plots saved")

# 7f. SHAP Force Plot (HTML)
force_html = shap.force_plot(
    explainer.expected_value,
    shap_values[high_risk_idx],
    X_test_df.iloc[high_risk_idx],
    feature_names=FEATURES,
    show=False
)
shap.save_html('plots/12_shap_force_plot.html', force_html)
print("  ✓ SHAP force plot (HTML) saved")

# 7c. SHAP Decision Plot
plt.figure(figsize=(10, 8), facecolor=DARK_BG)
sample_indices = np.random.choice(len(shap_values), 50, replace=False)
shap.decision_plot(explainer.expected_value,
                   shap_values[sample_indices],
                   X_test_df.iloc[sample_indices],
                   feature_names=FEATURES,
                   show=False)
plt.title('SHAP Decision Plot — 50 Patient Predictions',
          fontsize=13, color=TEXT, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/13_shap_decision_plot.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ SHAP decision plot saved")

# ─── 8. Save Model Artifacts ─────────────────────────────────────────────────
print("\n[SAVE] Saving model artifacts...")

joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(rf_model,  'models/rf_model.pkl')
joblib.dump(scaler,    'models/scaler.pkl')
joblib.dump(FEATURES,  'models/features.pkl')

# Save SHAP values for web app
np.save('models/shap_values.npy', shap_values)
np.save('models/expected_value.npy', np.array([explainer.expected_value]))
X_test_df.to_csv('models/X_test.csv', index=False)
np.save('models/y_test.npy', y_test.values)
np.save('models/y_prob_xgb.npy', y_prob_xgb)

print("  ✓ All artifacts saved to models/")

# ─── 9. Summary Stats ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SHAP FEATURE IMPORTANCE RANKING")
print("=" * 60)
feat_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Mean |SHAP|': np.abs(shap_values).mean(0)
}).sort_values('Mean |SHAP|', ascending=False)

for i, row in feat_importance.iterrows():
    bar = '█' * int(row['Mean |SHAP|'] * 50)
    print(f"  {row['Feature']:25s} {bar} {row['Mean |SHAP|']:.4f}")

print("\n[DONE] All plots saved to plots/  |  Models saved to models/")
print("=" * 60)
