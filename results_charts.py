# python
import os
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional seaborn for nicer visuals
try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    sns = None

def _ensure_out_dir(base_out=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_out:
        out = Path(base_out) / ts
    else:
        out = Path("results") / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def _safe_get(d, *keys, default=np.nan):
    # nested get helper
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k, None) if isinstance(cur, dict) else None
    return cur if cur is not None else default

def plot_regression_summary(final_results, out_dir):
    rows = []
    for sector, models in final_results.get('regression', {}).items():
        for model_name, metrics in (models or {}).items():
            rows.append({
                "sector": sector,
                "model": model_name,
                "R2": _safe_get(metrics, 'R2', default=np.nan),
                "RMSE": _safe_get(metrics, 'RMSE', default=np.nan),
                "MSE": _safe_get(metrics, 'MSE', default=np.nan)
            })
    if not rows:
        return
    df = pd.DataFrame(rows).dropna(subset=["model"])
    if df.empty:
        return

    # Bar plots: RMSE and R2 grouped by sector
    for metric in ["RMSE", "R2"]:
        plt.figure(figsize=(10, 6))
        if sns:
            sns.barplot(data=df, x="sector", y=metric, hue="model", ci=None)
        else:
            for i, (sector, group) in enumerate(df.groupby("sector")):
                x = np.arange(len(group))
                width = 0.8 / len(group["model"].unique())
            df_pivot = df.pivot_table(index="sector", columns="model", values=metric)
            df_pivot.plot(kind="bar", figsize=(10,6))
        plt.title(f"Regression - {metric} by Sector & Model")
        plt.tight_layout()
        plt.savefig(out_dir / f"regression_{metric.lower()}.png", dpi=150)
        plt.close()

    # Heatmap of models vs sectors for RMSE (normalized)
    pivot = df.pivot_table(index="model", columns="sector", values="RMSE")
    if not pivot.empty:
        plt.figure(figsize=(8, max(3, pivot.shape[0]*0.6)))
        if sns:
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
        else:
            plt.imshow(pivot.fillna(pivot.mean().mean()), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        plt.title("Heatmap - RMSE (model vs sector)")
        plt.tight_layout()
        plt.savefig(out_dir / "regression_rmse_heatmap.png", dpi=150)
        plt.close()

def plot_classification_binary_summary(final_results, out_dir):
    rows = []
    for sector, models in final_results.get('classification_binary', {}).items():
        for model_name, metrics in (models or {}).items():
            dirs = metrics.get('Direction', {}) if isinstance(metrics, dict) else {}
            total = sum(dirs.values()) or 1
            down_pct = dirs.get('Down', 0) / total * 100
            up_pct = dirs.get('Up', 0) / total * 100
            rows.append({
                "sector": sector,
                "model": model_name,
                "F1": _safe_get(metrics, 'F1', default=np.nan),
                "Precision": _safe_get(metrics, 'Precision', default=np.nan),
                "Down%": down_pct,
                "Up%": up_pct
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # F1 and Precision bar charts
    for metric in ["F1", "Precision"]:
        plt.figure(figsize=(10, 6))
        if sns:
            sns.barplot(data=df, x="sector", y=metric, hue="model", ci=None)
        else:
            df_pivot = df.pivot_table(index="sector", columns="model", values=metric)
            df_pivot.plot(kind="bar", figsize=(10,6))
        plt.title(f"Binary Classification - {metric} by Sector & Model")
        plt.tight_layout()
        plt.savefig(out_dir / f"class_binary_{metric.lower()}.png", dpi=150)
        plt.close()

    # Stacked bar for Down% vs Up%
    pivot = df.pivot_table(index=["sector","model"], values=["Down%","Up%"]).reset_index()
    for sector, group in pivot.groupby("sector"):
        plt.figure(figsize=(8, max(4, len(group)*0.5)))
        idx = np.arange(len(group))
        down = group["Down%"].values
        up = group["Up%"].values
        plt.bar(idx, down, label="Down %", color="#d9534f")
        plt.bar(idx, up, bottom=down, label="Up %", color="#5cb85c")
        plt.xticks(idx, group["model"], rotation=45, ha="right")
        plt.ylabel("Percentage")
        plt.title(f"Direction % - {sector}")
        plt.legend()
        plt.tight_layout()
        safe_name = sector.lower().replace(" ", "_")
        plt.savefig(out_dir / f"class_binary_direction_{safe_name}.png", dpi=150)
        plt.close()

def plot_multi_class_summary(final_results, out_dir):
    rows = []
    for sector, models in final_results.get('classification_multi', {}).items():
        for model_name, metrics in (models or {}).items():
            dirs = metrics.get('Direction', {}) if isinstance(metrics, dict) else {}
            total = sum(dirs.values()) or 1
            down = dirs.get('Down',0)/total*100
            neu = dirs.get('Neutral',0)/total*100
            up = dirs.get('Up',0)/total*100
            rows.append({
                "sector": sector,
                "model": model_name,
                "F1": _safe_get(metrics, 'F1', default=np.nan),
                "Precision": _safe_get(metrics, 'Precision', default=np.nan),
                "Down%": down, "Neutral%": neu, "Up%": up
            })
    if not rows:
        return
    df = pd.DataFrame(rows)

    # F1 chart
    plt.figure(figsize=(10,6))
    if sns:
        sns.barplot(data=df, x="sector", y="F1", hue="model", ci=None)
    else:
        df_pivot = df.pivot_table(index="sector", columns="model", values="F1")
        df_pivot.plot(kind="bar", figsize=(10,6))
    plt.title("Multi-class Classification - F1 by Sector & Model")
    plt.tight_layout()
    plt.savefig(out_dir / "class_multi_f1.png", dpi=150)
    plt.close()

def plot_comparison_radar(final_results, out_dir):
    """
    Creates a radar-like comparison for top models across scopes.
    Normalizes metrics per-scope to [0,1] and plots radar per model.
    """
    # collect average scores per model across scopes
    scopes = ['regression', 'classification_binary', 'classification_multi']
    model_scores = {}
    metrics_map = {
        'regression': ('RMSE', 'R2'),
        'classification_binary': ('F1', 'Precision'),
        'classification_multi': ('F1', 'Precision')
    }
    for scope in scopes:
        scope_dict = final_results.get(scope, {}) or {}
        for sector, models in scope_dict.items():
            for model_name, metrics in (models or {}).items():
                if model_name not in model_scores:
                    model_scores[model_name] = {}
                for metric in metrics_map.get(scope, ()):
                    val = _safe_get(metrics, metric, default=np.nan)
                    # for RMSE lower is better -> invert later
                    model_scores[model_name][f"{scope}_{metric}"] = val

    if not model_scores:
        return

    # Build DataFrame
    df = pd.DataFrame.from_dict(model_scores, orient='index').fillna(np.nan)
    if df.empty:
        return

    # Normalize each column to 0-1 (for RMSE invert so higher is better)
    norm_df = df.copy()
    for col in norm_df.columns:
        col_vals = norm_df[col].values.astype(float)
        # if it's RMSE column invert
        if "RMSE" in col:
            col_vals = -col_vals
        mn = np.nanmin(col_vals)
        mx = np.nanmax(col_vals)
        if np.isfinite(mn) and np.isfinite(mx) and mx != mn:
            norm_df[col] = (col_vals - mn) / (mx - mn)
        else:
            norm_df[col] = 0.5  # neutral

    # Plot small radar-like spider charts (one per top-N models)
    top_models = norm_df.mean(axis=1).sort_values(ascending=False).head(6).index
    labels = list(norm_df.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for model in top_models:
        values = norm_df.loc[model].tolist()
        values += values[:1]
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
        ax.set_title(f"Model comparison radar - {model}")
        ax.set_ylim(0,1)
        plt.tight_layout()
        safe = model.lower().replace(" ", "_")
        plt.savefig(out_dir / f"radar_{safe}.png", dpi=150)
        plt.close()

def save_all_charts(final_results, detailed_results=None, base_out=None):
    """
    Main entrypoint. final_results is expected to match the structure produced
    in the main script: {'regression': {sector: {model: metrics}} , 'classification_binary': {...}, ...}
    detailed_results is optional and can be used for yearwise plots if provided
    (structure must be provided by caller).
    """
    out_dir = _ensure_out_dir(base_out)
    # Save a copy of the raw final_results for traceability
    try:
        with open(out_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, default=str, indent=2)
    except Exception:
        pass

    plot_regression_summary(final_results, out_dir)
    plot_classification_binary_summary(final_results, out_dir)
    plot_multi_class_summary(final_results, out_dir)
    plot_comparison_radar(final_results, out_dir)

    # If detailed/yearwise results provided, try plotting trends
    if detailed_results:
        try:
            # expected: detailed_results[sector][model] -> list of per-year dicts with 'year' and metrics
            for sector, models in detailed_results.items():
                for model_name, per_year in (models or {}).items():
                    df = pd.DataFrame(per_year)
                    if df.empty:
                        continue
                    df = df.sort_values("year")
                    plt.figure(figsize=(10,4))
                    for metric in ["R2","RMSE","F1","Precision","MSE"]:
                        if metric in df.columns:
                            plt.plot(df["year"], df[metric], marker="o", label=metric)
                    plt.title(f"Yearwise metrics - {sector} - {model_name}")
                    plt.xlabel("Year")
                    plt.legend()
                    plt.tight_layout()
                    safe_sector = sector.lower().replace(" ", "_")
                    safe_model = model_name.lower().replace(" ", "_")
                    plt.savefig(out_dir / f"yearwise_{safe_sector}_{safe_model}.png", dpi=150)
                    plt.close()
        except Exception:
            pass

    return out_dir

# CLI helper so this file can be run standalone after model run:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create charts from final_results JSON")
    parser.add_argument("--results", type=str, help="Path to final_results JSON produced by main script", required=False)
    parser.add_argument("--detailed", type=str, help="Optional detailed/yearwise JSON", required=False)
    parser.add_argument("--out", type=str, help="Optional base output directory", required=False)
    args = parser.parse_args()

    final = {}
    detailed = None
    if args.results and Path(args.results).exists():
        try:
            final = json.loads(Path(args.results).read_text())
        except Exception:
            final = {}
    if args.detailed and Path(args.detailed).exists():
        try:
            detailed = json.loads(Path(args.detailed).read_text())
        except Exception:
            detailed = None

    out = save_all_charts(final, detailed_results=detailed, base_out=args.out)
    print(f"Charts written to: {out}")
