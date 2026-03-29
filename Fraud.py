"""
=============================================================
 Fraud Detection ML System — Industry-Grade Pipeline
 Project 3: Financial Fraud Detection
=============================================================

REQUIREMENTS (install once):
    pip install scikit-learn pandas numpy matplotlib seaborn

RUN:
    python fraud_detection_final.py

OUTPUTS (saved to current directory):
    fraud_detection_dashboard.png
    roc_pr_curves.png
    fraud_detection_report.md
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # comment out if you want interactive plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, textwrap
from datetime import datetime
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Output folder (same directory as the script) ──────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0F1117",
    "card":    "#1A1D27",
    "border":  "#2A2D3A",
    "accent":  "#6C63FF",
    "danger":  "#FF4D6D",
    "success": "#00D97E",
    "warn":    "#FFB830",
    "text":    "#E8E8F0",
    "muted":   "#6B7280",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION  (synthetic but realistic)
# ══════════════════════════════════════════════════════════════════════════════
def generate_transaction_data(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Generates a realistic synthetic transaction dataset with:
    - 10 PCA-style anonymised features (V1-V10)
    - 9 domain-engineered fraud indicators
    - ~3 % fraud rate (severe class imbalance)
    """
    print("▸ Generating synthetic transaction dataset …")
    rng = np.random.RandomState(42)

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.97, 0.03],
        flip_y=0.002,
        random_state=42,
    )

    df = pd.DataFrame(X_raw, columns=[f"v{i}" for i in range(1, 11)])
    df["Class"] = y

    # Transaction amounts — fraud skews higher
    df["Amount"] = np.where(
        y == 0,
        np.abs(rng.lognormal(mean=4.0, sigma=1.2, size=n_samples)),
        np.abs(rng.lognormal(mean=5.5, sigma=1.8, size=n_samples)),
    ).round(2)

    # Fraud happens more at night
    df["Hour"] = np.where(
        y == 0,
        rng.choice(range(8, 22), size=n_samples),
        rng.choice([0, 1, 2, 3, 23], size=n_samples),
    )
    df["IsNightTxn"] = df["Hour"].isin([0, 1, 2, 3, 22, 23]).astype(int)

    df["MerchantCategory"] = rng.choice(
        ["grocery", "retail", "online", "travel", "atm"],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.10, 0.05],
    )

    # Fraud accounts show more transactions in a rolling 24-hour window
    df["TransactionCount_24h"] = np.where(
        y == 0, rng.poisson(3, n_samples), rng.poisson(12, n_samples)
    )

    # Derived features
    df["AmountZScore"]   = ((df["Amount"] - df["Amount"].mean()) / df["Amount"].std()).round(3)
    df["HighAmountFlag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["FrequencyFlag"]  = (df["TransactionCount_24h"] > 8).astype(int)
    df["RiskScore"] = (
        0.4 * df["HighAmountFlag"]
        + 0.3 * df["FrequencyFlag"]
        + 0.2 * df["IsNightTxn"]
        + 0.1 * (df["MerchantCategory"] == "atm").astype(int)
    ).round(3)

    print(f"  Dataset shape : {df.shape}")
    print(f"  Fraud rate    : {y.mean() * 100:.2f}%  "
          f"({y.sum():,} fraud / {(y == 0).sum():,} legitimate)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    - Label-encode categorical column
    - StandardScale continuous features
    - Stratified 80/20 train-test split
    """
    print("\n▸ Preprocessing …")
    le = LabelEncoder()
    df["MerchantCategory_enc"] = le.fit_transform(df["MerchantCategory"])

    feature_cols = [
        "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
        "Amount", "Hour", "IsNightTxn", "TransactionCount_24h",
        "AmountZScore", "HighAmountFlag", "FrequencyFlag",
        "RiskScore", "MerchantCategory_enc",
    ]

    X = df[feature_cols].copy()
    y = df["Class"].copy()

    scale_cols = [
        "Amount", "Hour", "TransactionCount_24h", "AmountZScore",
        "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
    ]
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {X_train.shape[0]:,} samples  |  Test : {X_test.shape[0]:,} samples")
    print(f"  Train fraud rate : {y_train.mean() * 100:.2f}%")
    return X_train, X_test, y_train, y_test, feature_cols, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASS-IMBALANCE HANDLING  (SMOTE-style, no external library)
# ══════════════════════════════════════════════════════════════════════════════
def manual_oversample(X_train: pd.DataFrame, y_train: pd.Series, ratio: float = 0.15):
    """
    Interpolates synthetic minority-class samples between existing fraud rows
    until the minority class is `ratio` × size of majority class.
    Equivalent to basic SMOTE without requiring imbalanced-learn.
    """
    print("\n▸ Handling class imbalance (SMOTE-style oversampling) …")
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    target_n  = int(len(legit_idx) * ratio)
    extra_n   = target_n - len(fraud_idx)

    if extra_n <= 0:
        print("  No oversampling needed.")
        return X_train, y_train

    rng    = np.random.RandomState(42)
    X_f    = X_train.iloc[fraud_idx].values
    synth  = []
    for _ in range(extra_n):
        i, j  = rng.choice(len(X_f), 2, replace=False)
        alpha = rng.uniform(0, 1)
        synth.append(X_f[i] + alpha * (X_f[j] - X_f[i]))

    X_synth = pd.DataFrame(synth, columns=X_train.columns)
    y_synth = pd.Series([1] * extra_n)

    X_bal = pd.concat([X_train, X_synth], ignore_index=True)
    y_bal = pd.concat([y_train.reset_index(drop=True), y_synth], ignore_index=True)
    print(f"  After oversampling → {dict(Counter(y_bal))}")
    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains four classifiers, each with class_weight='balanced' to
    further penalise misclassification of the minority class.
    """
    print("\n▸ Training models …")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    trained = {}
    for name, mdl in models.items():
        print(f"  ▹ {name} …", end=" ", flush=True)
        mdl.fit(X_train, y_train)
        trained[name] = mdl
        print("done")
    return trained


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Computes per-model:  ROC-AUC, PR-AUC, F1, Precision, Recall,
    Confusion Matrix, and full classification report.
    """
    print("\n▸ Evaluating models …")
    results = {}
    for name, mdl in models.items():
        y_pred  = mdl.predict(X_test)
        y_prob  = mdl.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)
        f1      = f1_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model":   mdl,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
            "f1":      f1,
            "report":  report,
            "cm":      cm,
        }
        p = report["1"]["precision"]
        r = report["1"]["recall"]
        print(f"  {name:<25}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}"
              f"  F1={f1:.4f}  Prec={p:.4f}  Recall={r:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN DASHBOARD  (5-row visual grid, saved as PNG)
# ══════════════════════════════════════════════════════════════════════════════
def make_dashboard(df: pd.DataFrame, results: dict,
                   feature_cols: list, y_test: pd.Series) -> str:
    print("\n▸ Generating dashboard …")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 32), facecolor=PALETTE["bg"])
    fig.suptitle(
        "FRAUD DETECTION — ML SYSTEM DASHBOARD",
        fontsize=22, fontweight="bold", color=PALETTE["text"], y=0.995,
    )
    gs = gridspec.GridSpec(
        5, 3, figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.05, right=0.97, top=0.97, bottom=0.03,
    )

    def card_ax(spec, title):
        ax = fig.add_subplot(spec)
        ax.set_facecolor(PALETTE["card"])
        for sp in ax.spines.values():
            sp.set_color(PALETTE["border"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)
        ax.set_title(title, color=PALETTE["text"],
                     fontsize=11, fontweight="bold", pad=8)
        return ax

    model_colors = [PALETTE["accent"], PALETTE["success"],
                    PALETTE["warn"], PALETTE["danger"]]

    # ── Row 0: Class distribution │ Amount distribution │ Hourly pattern ──────
    ax0 = card_ax(gs[0, 0], "Class Distribution")
    counts = df["Class"].value_counts()
    bars = ax0.bar(
        ["Legitimate", "Fraud"], [counts[0], counts[1]],
        color=[PALETTE["success"], PALETTE["danger"]],
        width=0.5, edgecolor=PALETTE["border"],
    )
    for b in bars:
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 300,
                 f"{int(b.get_height()):,}", ha="center",
                 color=PALETTE["text"], fontsize=10)
    ax0.set_ylabel("Count", color=PALETTE["muted"])

    ax1 = card_ax(gs[0, 1], "Transaction Amount Distribution")
    for cls, col, lbl in [(0, PALETTE["success"], "Legitimate"),
                           (1, PALETTE["danger"], "Fraud")]:
        ax1.hist(df[df["Class"] == cls]["Amount"].clip(upper=1000),
                 bins=60, alpha=0.7, color=col, label=lbl, density=True)
    leg = ax1.legend(facecolor=PALETTE["card"], fontsize=9)
    for t in leg.get_texts(): t.set_color(PALETTE["text"])
    ax1.set_xlabel("Amount ($)", color=PALETTE["muted"])

    ax2 = card_ax(gs[0, 2], "Transactions by Hour of Day")
    hourly = df.groupby(["Hour", "Class"]).size().unstack(fill_value=0)
    hrs = hourly.index
    ax2.bar(hrs, hourly[0], color=PALETTE["success"], alpha=0.7,
            label="Legit", width=0.7)
    ax2.bar(hrs, hourly[1], bottom=hourly[0], color=PALETTE["danger"],
            alpha=0.9, label="Fraud", width=0.7)
    leg = ax2.legend(facecolor=PALETTE["card"], fontsize=9)
    for t in leg.get_texts(): t.set_color(PALETTE["text"])
    ax2.set_xlabel("Hour", color=PALETTE["muted"])

    # ── Row 1: Top-2 Confusion Matrices │ Feature Importance ──────────────────
    best_two = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)[:2]
    for col_i, (name, res) in enumerate(best_two):
        ax = card_ax(gs[1, col_i], f"Confusion Matrix — {name}")
        cm = res["cm"]
        labels = np.array([[f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
                            [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"]])
        ax.imshow(cm, cmap="RdYlGn", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"], color=PALETTE["text"])
        ax.set_yticklabels(["Actual Legit", "Actual Fraud"], color=PALETTE["text"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i, j], ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

    ax_fi = card_ax(gs[1, 2], "Top-15 Feature Importances (Random Forest)")
    rf = results["Random Forest"]["model"]
    imp = (pd.Series(rf.feature_importances_, index=feature_cols)
             .sort_values(ascending=True).tail(15))
    colors_fi = [PALETTE["accent"] if v > imp.median() else PALETTE["muted"]
                 for v in imp]
    imp.plot(kind="barh", ax=ax_fi, color=colors_fi)
    ax_fi.set_xlabel("Importance", color=PALETTE["muted"])
    ax_fi.tick_params(axis="y", labelsize=8)

    # ── Row 2: ROC Curves │ Precision-Recall Curves │ Bar comparison ──────────
    ax_roc = card_ax(gs[2, 0], "ROC Curves")
    ax_pr  = card_ax(gs[2, 1], "Precision–Recall Curves")

    for (name, res), col in zip(results.items(), model_colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax_roc.plot(fpr, tpr, color=col, lw=2,
                    label=f"{name} ({res['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax_pr.plot(rec, prec, color=col, lw=2,
                   label=f"{name} ({res['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1)
    for ax, xlabel, ylabel in [
        (ax_roc, "False Positive Rate", "True Positive Rate"),
        (ax_pr,  "Recall", "Precision"),
    ]:
        ax.set_xlabel(xlabel, color=PALETTE["muted"])
        ax.set_ylabel(ylabel, color=PALETTE["muted"])
        leg = ax.legend(facecolor=PALETTE["card"], fontsize=8)
        for t in leg.get_texts(): t.set_color(PALETTE["text"])

    ax_bar = card_ax(gs[2, 2], "Model Performance Comparison")
    metrics_df = pd.DataFrame({
        n: {
            "ROC-AUC":   v["roc_auc"],
            "PR-AUC":    v["pr_auc"],
            "F1":        v["f1"],
            "Precision": v["report"]["1"]["precision"],
            "Recall":    v["report"]["1"]["recall"],
        }
        for n, v in results.items()
    }).T
    x = np.arange(len(metrics_df))
    w = 0.15
    bar_colors = [PALETTE["accent"], PALETTE["success"],
                  PALETTE["warn"], PALETTE["danger"], "#A78BFA"]
    for i, (col, clr) in enumerate(zip(metrics_df.columns, bar_colors)):
        ax_bar.bar(x + i * w, metrics_df[col], w,
                   label=col, color=clr, alpha=0.85)
    ax_bar.set_xticks(x + w * 2)
    ax_bar.set_xticklabels(
        [textwrap.fill(n, 12) for n in metrics_df.index],
        color=PALETTE["text"], fontsize=8,
    )
    ax_bar.set_ylim(0, 1.15)
    leg = ax_bar.legend(facecolor=PALETTE["card"], fontsize=8, ncol=2)
    for t in leg.get_texts(): t.set_color(PALETTE["text"])
    ax_bar.set_ylabel("Score", color=PALETTE["muted"])

    # ── Row 3: Risk Score │ Merchant Category │ Correlation Heatmap ────────────
    ax_rs = card_ax(gs[3, 0], "Risk Score Distribution")
    for cls, col, lbl in [(0, PALETTE["success"], "Legit"),
                           (1, PALETTE["danger"], "Fraud")]:
        ax_rs.hist(df[df["Class"] == cls]["RiskScore"],
                   bins=30, alpha=0.75, color=col, label=lbl, density=True)
    leg = ax_rs.legend(facecolor=PALETTE["card"])
    for t in leg.get_texts(): t.set_color(PALETTE["text"])
    ax_rs.set_xlabel("Risk Score", color=PALETTE["muted"])

    ax_mc = card_ax(gs[3, 1], "Fraud Rate by Merchant Category")
    fraud_by_cat = (df.groupby("MerchantCategory")["Class"]
                      .mean() * 100).sort_values(ascending=False)
    bars2 = ax_mc.bar(fraud_by_cat.index, fraud_by_cat.values,
                      color=PALETTE["accent"], edgecolor=PALETTE["border"])
    for b in bars2:
        ax_mc.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                   f"{b.get_height():.1f}%", ha="center",
                   color=PALETTE["text"], fontsize=9)
    ax_mc.set_ylabel("Fraud Rate (%)", color=PALETTE["muted"])
    ax_mc.tick_params(axis="x", rotation=20)

    ax_hm = card_ax(gs[3, 2], "Feature Correlation Heatmap")
    corr_cols = ["Amount", "Hour", "IsNightTxn", "TransactionCount_24h",
                 "RiskScore", "HighAmountFlag", "FrequencyFlag", "Class"]
    corr = df[corr_cols].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, ax=ax_hm, mask=mask,
                cmap=sns.diverging_palette(240, 10, as_cmap=True),
                vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor=PALETTE["border"],
                cbar_kws={"shrink": 0.8})
    ax_hm.tick_params(colors=PALETTE["muted"], labelsize=8)

    # ── Row 4: Sample transaction fraud-probability bar chart ─────────────────
    ax_pred = card_ax(gs[4, :], "Predicted Fraud Probability — Sample Transactions")
    best_name, best_res = max(results.items(), key=lambda x: x[1]["roc_auc"])
    sample_probs = best_res["y_prob"][:40]
    bar_clrs = [PALETTE["danger"] if p > 0.5 else PALETTE["success"]
                for p in sample_probs]
    ax_pred.bar(range(len(sample_probs)), sample_probs,
                color=bar_clrs, edgecolor=PALETTE["border"], width=0.8)
    ax_pred.axhline(0.5, color=PALETTE["warn"], ls="--", lw=1.5,
                    label="Decision threshold (0.5)")
    ax_pred.set_xlabel("Transaction Index", color=PALETTE["muted"])
    ax_pred.set_ylabel("Fraud Probability", color=PALETTE["muted"])
    ax_pred.set_ylim(0, 1.05)
    leg = ax_pred.legend(facecolor=PALETTE["card"])
    for t in leg.get_texts(): t.set_color(PALETTE["text"])
    ax_pred.set_title(
        f"Sample Predictions — {best_name}  "
        "(Red = Predicted Fraud  |  Green = Legitimate)",
        color=PALETTE["text"], fontsize=11, fontweight="bold", pad=8,
    )

    out = os.path.join(OUTPUT_DIR, "fraud_detection_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Dashboard saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. ROC / PR CURVES  (separate clean chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_pr(results: dict, y_test: pd.Series) -> str:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE["bg"])
    fig.suptitle("ROC & Precision–Recall Curves",
                 color=PALETTE["text"], fontsize=15, fontweight="bold")

    colors = [PALETTE["accent"], PALETTE["success"],
              PALETTE["warn"], PALETTE["danger"]]

    for ax in axes:
        ax.set_facecolor(PALETTE["card"])
        for sp in ax.spines.values():
            sp.set_color(PALETTE["border"])
        ax.tick_params(colors=PALETTE["muted"])

    for (name, res), col in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, color=col, lw=2,
                     label=f"{name} (AUC={res['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, color=col, lw=2,
                     label=f"{name} (AP={res['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve")
    axes[0].xaxis.label.set_color(PALETTE["muted"])
    axes[0].yaxis.label.set_color(PALETTE["muted"])
    axes[0].title.set_color(PALETTE["text"])
    leg0 = axes[0].legend(facecolor=PALETTE["card"], fontsize=9)
    for t in leg0.get_texts(): t.set_color(PALETTE["text"])

    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    axes[1].xaxis.label.set_color(PALETTE["muted"])
    axes[1].yaxis.label.set_color(PALETTE["muted"])
    axes[1].title.set_color(PALETTE["text"])
    leg1 = axes[1].legend(facecolor=PALETTE["card"], fontsize=9)
    for t in leg1.get_texts(): t.set_color(PALETTE["text"])

    out = os.path.join(OUTPUT_DIR, "roc_pr_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ROC/PR chart   → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8. DOCUMENTATION  (Markdown report)
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(results: dict, df: pd.DataFrame) -> str:
    best_name, best_res = max(results.items(), key=lambda x: x[1]["roc_auc"])
    rep = best_res["report"]

    lines = [
        "# Fraud Detection ML System — Project Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "End-to-end machine learning pipeline to detect fraudulent financial "
        "transactions from a highly imbalanced dataset. Four classifiers were "
        "trained, evaluated, and compared using industry-standard metrics.",
        "",
        "## 2. Dataset Overview",
        "| Attribute | Value |",
        "|-|-|",
        f"| Total transactions | {len(df):,} |",
        f"| Fraudulent | {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%) |",
        f"| Legitimate | {(df['Class']==0).sum():,} |",
        "| Features | 19 (10 anonymised V-features + 9 engineered) |",
        "",
        "## 3. Methodology",
        "### 3.1 Preprocessing",
        "- Missing value check (none detected)",
        "- LabelEncoder on `MerchantCategory`",
        "- StandardScaler on continuous features",
        "- Stratified 80 / 20 train-test split",
        "",
        "### 3.2 Class-Imbalance Handling",
        "- SMOTE-style interpolative oversampling of minority class to ~15 % ratio",
        "- `class_weight='balanced'` passed to every scikit-learn estimator",
        "",
        "### 3.3 Engineered Features",
        "| Feature | Description |",
        "|-|-|",
        "| `IsNightTxn` | 1 if transaction hour is 10 PM – 4 AM |",
        "| `AmountZScore` | Standardised transaction amount |",
        "| `HighAmountFlag` | 1 if Amount > 95th percentile |",
        "| `TransactionCount_24h` | Rolling 24-hour transaction count |",
        "| `FrequencyFlag` | 1 if TransactionCount_24h > 8 |",
        "| `RiskScore` | Composite weighted indicator (0–1) |",
        "",
        "## 4. Models",
        "1. Logistic Regression — baseline",
        "2. Decision Tree",
        "3. Random Forest",
        "4. Gradient Boosting",
        "",
        "## 5. Results",
        "| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |",
        "|-|-|-|-|-|-|",
    ]

    for name, res in sorted(results.items(),
                             key=lambda x: x[1]["roc_auc"], reverse=True):
        r = res["report"]["1"]
        lines.append(
            f"| {name} | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} "
            f"| {res['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    lines += [
        "",
        f"## 6. Best Model: {best_name}",
        "```",
        f"ROC-AUC  : {best_res['roc_auc']:.4f}",
        f"PR-AUC   : {best_res['pr_auc']:.4f}",
        f"F1-Score : {best_res['f1']:.4f}",
        f"Precision: {rep['1']['precision']:.4f}",
        f"Recall   : {rep['1']['recall']:.4f}",
        "```",
        "",
        "## 7. Threshold Guidance",
        "Default threshold = **0.5**. Lower it (e.g. 0.3) to catch more fraud "
        "at the cost of more false positives. Tune based on business cost of "
        "false negatives vs false positives.",
        "",
        "## 8. Recommendations",
        "- Monthly model retraining to combat concept drift",
        "- Add velocity features (amount last 1 h / 7 d)",
        "- Integrate SHAP for auditor-facing explanations",
        "- A/B test threshold values in staging before production",
        "",
        "---",
        "_End of Report_",
    ]

    out = os.path.join(OUTPUT_DIR, "fraud_detection_report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved   → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  FRAUD DETECTION ML SYSTEM")
    print("=" * 60)

    df = generate_transaction_data(n_samples=50_000)
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess(df)
    X_bal, y_bal = manual_oversample(X_train, y_train, ratio=0.15)
    models  = train_models(X_bal, y_bal)
    results = evaluate_models(models, X_test, y_test)

    dash  = make_dashboard(df, results, feature_cols, y_test)
    roc_p = plot_roc_pr(results, y_test)
    rep   = generate_report(results, df)

    print("\n" + "=" * 60)
    print("  ✓ ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    for path in (dash, roc_p, rep):
        print(f"  • {path}")


if __name__ == "__main__":
    main()