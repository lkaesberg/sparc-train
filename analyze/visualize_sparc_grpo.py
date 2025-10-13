import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn optional
    sns = None

import matplotlib.pyplot as plt


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def standardize_model_size_from_filename(filename: str) -> str:
    lower = filename.lower()
    # Capture Qwen3 model sizes like 0.6b, 1.7b, 4b, 8b, 14b, 32b
    match = re.search(r"qwen3[-_]?([0-9]+(?:\.[0-9]+)?b)", lower)
    if match:
        size = match.group(1).upper()
        return f"Qwen3-{size}"
    # Fallback: try to capture '-(\d+(?:.\d+)?B)-SPaRC' style
    match = re.search(r"qwen3[-_]?([0-9]+(?:\.[0-9]+)?)[-_]?b", lower)
    if match:
        size = f"{match.group(1).upper()}B"
        return f"Qwen3-{size}"
    # Last resort: keep original (but this should not happen for Qwen3)
    return "Unknown"


def extract_variant_from_filename(filename: str) -> str:
    # Variants are encoded like ...-GRPO-16R_stats.csv, ...-GRPO-8E_stats.csv, ...-GRPO-L_stats.csv
    lower = filename.lower()
    if "-grpo-16r" in lower:
        return "16R"
    if "-grpo-8e" in lower:
        return "8E"
    if "-grpo-l" in lower:
        return "L"
    # default GRPO (no suffix)
    if "-grpo" in lower:
        return "Default"
    # Non-GRPO file treated as baseline
    return "Baseline"


def parse_percentage(value: str) -> float:
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_stats_csv(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    # Normalize column names and metric values to handle minor variations
    df.columns = [c.strip() for c in df.columns]
    if "Metric" not in df.columns or "Percentage" not in df.columns:
        raise ValueError(f"Unexpected schema in {csv_path}")

    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Percentage"] = df["Percentage"].apply(parse_percentage)

    metrics = {m: None for m in [
        "Correctly Solved",
        "Difficulty 1 Solved",
        "Difficulty 2 Solved",
        "Difficulty 3 Solved",
        "Difficulty 4 Solved",
        "Difficulty 5 Solved",
        "Fully Valid Paths",
        "Connected Paths",
        "Correct Start/End",
        "Non-Intersecting",
        "No Rule Violations",
    ]}

    available = {row["Metric"]: row["Percentage"] for _, row in df.iterrows() if not pd.isna(row["Percentage"])}

    results: Dict[str, float] = {}
    for key in metrics.keys():
        if key in available:
            results[key] = float(available[key])

    return results


def collect_runs(input_dir: Path) -> List[Tuple[str, str, Path]]:
    csv_files = sorted(input_dir.glob("*_stats.csv"))
    runs: List[Tuple[str, str, Path]] = []
    excluded_models = {"Qwen3-1.7B", "Qwen3-8B"}
    for path in csv_files:
        name = path.name
        lower = name.lower()
        
        # Special handling for o4-mini
        if "o4-mini" in lower or "o4_mini" in lower:
            runs.append(("o4-mini (OpenAI)", "Baseline", path))
            continue
        
        # Restrict to Qwen3 family to avoid mixing other base models
        if "qwen3" not in lower:
            continue
        model = standardize_model_size_from_filename(name)
        if model in excluded_models:
            continue
        variant = extract_variant_from_filename(name)
        runs.append((model, variant, path))
    return runs


def build_dataframes(runs: List[Tuple[str, str, Path]]):
    overall_rows = []
    diff_rows = []
    err_rows = []

    for model, variant, path in runs:
        try:
            stats = parse_stats_csv(path)
        except Exception:
            continue

        overall = stats.get("Correctly Solved")
        if overall is not None:
            overall_rows.append({
                "model": model,
                "variant": variant,
                "accuracy": overall,
            })

        for d in range(1, 6):
            k = f"Difficulty {d} Solved"
            v = stats.get(k)
            if v is not None:
                diff_rows.append({
                    "model": model,
                    "variant": variant,
                    "difficulty": d,
                    "accuracy": v,
                })

        # Error rates = 100 - percentage satisfied, relabeled to the negative condition
        def metric_to_error_label(metric: str) -> str:
            mapping = {
                "Fully Valid Paths": "Invalid Paths",
                "Connected Paths": "Disconnected Paths",
                "Correct Start/End": "Wrong Start/End",
                "Non-Intersecting": "Intersecting Paths",
                "No Rule Violations": "Rule Violations",
            }
            return mapping.get(metric, metric)

        # Error rates = 100 - percentage satisfied
        for err_key in [
            "Fully Valid Paths",
            "Connected Paths",
            "Correct Start/End",
            "Non-Intersecting",
            "No Rule Violations",
        ]:
            v = stats.get(err_key)
            if v is not None:
                err_rows.append({
                    "model": model,
                    "variant": variant,
                    "error_type": metric_to_error_label(err_key),
                    "error_rate": max(0.0, 100.0 - v),
                })

    overall_df = pd.DataFrame(overall_rows)
    diff_df = pd.DataFrame(diff_rows)
    err_df = pd.DataFrame(err_rows)

    return overall_df, diff_df, err_df


def sort_categoricals(overall_df: pd.DataFrame, diff_df: pd.DataFrame, err_df: pd.DataFrame):
    full_model_order = [
        "o4-mini (OpenAI)",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]
    # Only include models actually present
    present_models = pd.unique(pd.concat([
        overall_df.get("model", pd.Series(dtype=str)),
        diff_df.get("model", pd.Series(dtype=str)),
        err_df.get("model", pd.Series(dtype=str)),
    ], ignore_index=True).dropna())
    model_order = [m for m in full_model_order if m in set(present_models)]

    variant_order = ["Baseline", "Default", "16R", "8E", "L"]

    def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            if "model" in df:
                df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
            if "variant" in df:
                df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
        return df

    return apply_categories(overall_df), apply_categories(diff_df), apply_categories(err_df)


def variant_to_label(variant: str) -> str:
    mapping = {
        "Baseline": "Baseline (no GRPO)",
        "Default": "GRPO: default (4 rollouts, 4 epochs)",
        "16R": "GRPO: 16 rollouts",
        "8E": "GRPO: 8 epochs",
        "L": "GRPO: low format reward",
    }
    return mapping.get(variant, variant)


def with_seaborn():
    return sns is not None


def style():
    if with_seaborn():
        sns.set_context("talk", font_scale=1.1)
        sns.set_style("whitegrid", {
            'axes.edgecolor': '.2',
            'grid.color': '.9',
            'grid.linestyle': '-',
            'axes.facecolor': 'white',
        })
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "semibold",
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })
    else:
        plt.style.use("ggplot")


def get_color_palette():
    """Professional color palette for variants"""
    return {
        "Baseline (no GRPO)": "#95a5a6",  # Cool gray
        "GRPO: default (4 rollouts, 4 epochs)": "#3498db",  # Bright blue
        "GRPO: 16 rollouts": "#e74c3c",  # Vibrant red
        "GRPO: 8 epochs": "#2ecc71",  # Fresh green
        "GRPO: low format reward": "#9b59b6",  # Rich purple
    }


def get_model_color_palette():
    """Professional color palette for different models (for summary plots)"""
    return {
        "o4-mini (OpenAI)": "#f39c12",  # Orange for OpenAI
        "Qwen3-0.6B": "#3498db",  # Blue
        "Qwen3-4B": "#2ecc71",  # Green
        "Qwen3-14B": "#e74c3c",  # Red
        "Qwen3-32B": "#9b59b6",  # Purple
    }


def get_palette_list(hue_order):
    """Get ordered color list from hue order"""
    pal = get_color_palette()
    return [pal.get(label, "#34495e") for label in hue_order]


def save_overall_accuracy_plot(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    style()
    
    # Prepare human-friendly labels
    df_plot = df.copy()
    df_plot["variant_label"] = df_plot["variant"].map(variant_to_label)
    if isinstance(df_plot["variant"].dtype, pd.CategoricalDtype):
        order_raw = [v for v in df_plot["variant"].cat.categories if (df_plot["variant"] == v).any()]
    else:
        order_raw = sorted(df_plot["variant"].unique())
    hue_order = [variant_to_label(v) for v in order_raw]
    palette = get_palette_list(hue_order)

    if with_seaborn():
        # Create a beautiful horizontal point plot with connected lines per variant
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Use pointplot for elegant connected markers
        sns.pointplot(
            data=df_plot,
            y="model",
            x="accuracy",
            hue="variant_label",
            hue_order=hue_order,
            palette=palette,
            markers=["o", "s", "D", "^", "v"][:len(hue_order)],
            linestyles=["-", "--", "-.", ":", "-"][:len(hue_order)],
            markersize=10,
            linewidth=2.5,
            dodge=0.4,
            errorbar=None,
            ax=ax,
        )
        
        ax.set_xlim(0, max(100, df_plot["accuracy"].max() * 1.1))
        ax.set_xlabel("Accuracy (%)", fontweight="bold")
        ax.set_ylabel("Model Size", fontweight="bold")
        ax.set_title("SPaRC GRPO: Overall Accuracy Comparison", pad=20, fontsize=18)
        
        # Add subtle grid
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Enhanced legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            title="Training Configuration",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
        )
    else:
        # Fallback grouped bars
        fig, ax = plt.subplots(figsize=(14, 7))
        models = df_plot["model"].cat.categories.tolist() if isinstance(df_plot["model"].dtype, pd.CategoricalDtype) else sorted(df_plot["model"].unique())
        variants = order_raw
        labels = hue_order
        x = range(len(models))
        width = 0.16 if len(variants) > 0 else 0.5
        for i, (var, label) in enumerate(zip(variants, labels)):
            sub = df_plot[df_plot["variant"] == var]
            vals = [sub[sub["model"] == m]["accuracy"].mean() if (sub["model"] == m).any() else float("nan") for m in models]
            ax.bar([xi + i * width for xi in x], vals, width=width, label=label, color=palette[i])
        center = ((len(variants) - 1) * width) / 2.0
        ax.set_xticks([xi + center for xi in x])
        ax.set_xticklabels(models, rotation=0)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Model")
        ax.set_title("SPaRC GRPO: Overall Accuracy by Variant")
        ax.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()


def save_difficulty_accuracy_plot(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    style()
    
    # Prepare human-friendly labels
    df_plot = df.copy()
    df_plot["variant_label"] = df_plot["variant"].map(variant_to_label)
    if isinstance(df_plot["variant"].dtype, pd.CategoricalDtype):
        order_raw = [v for v in df_plot["variant"].cat.categories if (df_plot["variant"] == v).any()]
    else:
        order_raw = sorted(df_plot["variant"].unique())
    hue_order = [variant_to_label(v) for v in order_raw]
    palette = get_palette_list(hue_order)

    if with_seaborn():
        # Beautiful line plot with facets
        g = sns.relplot(
            data=df_plot,
            kind="line",
            x="difficulty",
            y="accuracy",
            hue="variant_label",
            hue_order=hue_order,
            palette=palette,
            col="model",
            col_wrap=2,
            height=4.5,
            aspect=1.3,
            markers=["o", "s", "D", "^", "v"][:len(hue_order)],
            dashes=False,
            markersize=10,
            linewidth=3,
            errorbar=None,
        )
        
        # Styling each subplot
        for ax in g.axes.flat:
            ax.set_ylim(0, 100)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            # Add subtle background shading for difficulty zones
            ax.axhspan(0, 20, alpha=0.05, color='red', zorder=0)
            ax.axhspan(20, 50, alpha=0.05, color='orange', zorder=0)
            ax.axhspan(50, 100, alpha=0.05, color='green', zorder=0)
        
        g.set_axis_labels("Difficulty Level", "Accuracy (%)", fontweight="bold")
        g.set_titles("{col_name}", fontweight="bold", size=14)
        
        # Enhanced legend
        g.fig.subplots_adjust(top=0.92, right=0.85)
        g.fig.suptitle("SPaRC GRPO: Accuracy Across Difficulty Levels", fontsize=18, fontweight="bold", y=0.98)
        
        if g._legend:
            g._legend.set_title("Training Configuration", prop={'weight': 'bold', 'size': 11})
            g._legend.set_bbox_to_anchor((1.02, 0.5))
            g._legend.set_loc("center left")
            g._legend.set_frame_on(True)
            g._legend.get_frame().set_facecolor('white')
            g._legend.get_frame().set_alpha(0.9)
            g._legend.get_frame().set_edgecolor('gray')
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
        plt.close()
    else:
        # Fallback
        models = df_plot["model"].cat.categories.tolist() if isinstance(df_plot["model"].dtype, pd.CategoricalDtype) else sorted(df_plot["model"].unique())
        n = len(models)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        variants = order_raw
        labels = hue_order
        for ax, m in zip(axes, models):
            sub = df_plot[df_plot["model"] == m]
            for i, (var, label) in enumerate(zip(variants, labels)):
                sub2 = sub[sub["variant"] == var]
                if sub2.empty:
                    continue
                ax.plot(sub2["difficulty"], sub2["accuracy"], marker="o", label=label, color=palette[i], linewidth=2)
            ax.set_title(str(m))
            ax.set_ylabel("Accuracy (%)")
            ax.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Difficulty")
        fig.suptitle("SPaRC GRPO: Accuracy by Difficulty and Variant")
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_error_rates_plot(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    style()
    
    # Prepare human-friendly labels
    df_plot = df.copy()
    df_plot["variant_label"] = df_plot["variant"].map(variant_to_label)
    if isinstance(df_plot["variant"].dtype, pd.CategoricalDtype):
        order_raw = [v for v in df_plot["variant"].cat.categories if (df_plot["variant"] == v).any()]
    else:
        order_raw = sorted(df_plot["variant"].unique())
    hue_order = [variant_to_label(v) for v in order_raw]
    
    # Get unique error types in consistent order
    error_types = [
        "Invalid Paths",
        "Disconnected Paths", 
        "Wrong Start/End",
        "Intersecting Paths",
        "Rule Violations",
    ]
    
    models = df_plot["model"].cat.categories.tolist() if isinstance(df_plot["model"].dtype, pd.CategoricalDtype) else sorted(df_plot["model"].unique())

    if with_seaborn():
        # Create elegant heatmap grid
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 6), sharey=False)
        if n_models == 1:
            axes = [axes]
        
        for idx, (ax, model) in enumerate(zip(axes, models)):
            # Prepare pivot table for this model
            model_data = df_plot[df_plot["model"] == model]
            pivot = model_data.pivot_table(
                index="error_type",
                columns="variant_label",
                values="error_rate",
                aggfunc="mean",
                observed=False
            )
            
            # For models with only baseline (like o4-mini), only use baseline column
            if model == "o4-mini (OpenAI)":
                available_variants = [v for v in hue_order if v in pivot.columns]
                pivot = pivot.reindex(index=error_types, columns=available_variants)
            else:
                # Reindex to ensure consistent ordering
                pivot = pivot.reindex(index=error_types, columns=hue_order)
            
            # Skip if no data
            if pivot.empty or pivot.isna().all().all():
                continue
            
            # Create heatmap with annotations
            sns.heatmap(
                pivot,
                ax=ax,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn_r',  # Red for high errors, green for low
                vmin=0,
                vmax=60,  # Adjust scale to accommodate o4-mini's higher errors
                cbar=idx == n_models - 1,  # Only show colorbar on last plot
                linewidths=1,
                linecolor='white',
                square=False,
                cbar_kws={'label': 'Error Rate (%)', 'shrink': 0.8} if idx == n_models - 1 else None,
                yticklabels=True,  # Always show y-axis labels
            )
            
            ax.set_title(f"{model}", fontsize=15, fontweight='bold', pad=12)
            ax.set_xlabel("Configuration", fontsize=12, fontweight='semibold')
            if idx == 0:
                ax.set_ylabel("Error Type", fontsize=12, fontweight='semibold')
            else:
                ax.set_ylabel("Error Type", fontsize=12, fontweight='semibold')
            
            # Rotate x labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        
        fig.suptitle("SPaRC GRPO: Error Rate Analysis (Lower is Better)", 
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
        plt.close()
    else:
        # Fallback bar plot
        n = len(models)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 4.5 * n), sharex=False)
        if n == 1:
            axes = [axes]
        variants = order_raw
        labels = hue_order
        palette = get_palette_list(hue_order)
        
        for ax, m in zip(axes, models):
            sub = df_plot[df_plot["model"] == m]
            x = range(len(error_types))
            width = 0.16 if len(variants) > 0 else 0.5
            for i, (var, label) in enumerate(zip(variants, labels)):
                sub2 = sub[sub["variant"] == var]
                vals = [sub2[sub2["error_type"] == et]["error_rate"].mean() if (sub2["error_type"] == et).any() else 0.0 for et in error_types]
                ax.bar([xi + i * width for xi in x], vals, width=width, label=label, color=palette[i])
            ax.set_title(str(m), fontweight='bold')
            center = ((len(variants) - 1) * width) / 2.0
            ax.set_xticks([xi + center for xi in x])
            ax.set_xticklabels(error_types, rotation=35, ha='right')
            ax.set_ylabel("Error Rate (%)")
            ax.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle("SPaRC GRPO: Error Rates by Variant")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_summary_dashboard(overall_df: pd.DataFrame, diff_df: pd.DataFrame, err_df: pd.DataFrame, out_path: Path):
    """Create a comprehensive summary dashboard with all key metrics"""
    if overall_df.empty or diff_df.empty or err_df.empty:
        return
    
    style()
    
    # Prepare data
    overall_plot = overall_df.copy()
    overall_plot["variant_label"] = overall_plot["variant"].map(variant_to_label)
    
    diff_plot = diff_df.copy()
    diff_plot["variant_label"] = diff_plot["variant"].map(variant_to_label)
    
    err_plot = err_df.copy()
    err_plot["variant_label"] = err_plot["variant"].map(variant_to_label)
    
    if isinstance(overall_plot["variant"].dtype, pd.CategoricalDtype):
        order_raw = [v for v in overall_plot["variant"].cat.categories if (overall_plot["variant"] == v).any()]
    else:
        order_raw = sorted(overall_plot["variant"].unique())
    hue_order = [variant_to_label(v) for v in order_raw]
    palette = get_palette_list(hue_order)
    
    if not with_seaborn():
        return  # Only create this for seaborn
    
    # Create 2x2 grid
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.92)
    
    # Top left: Overall accuracy by model
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Filter to only include variant/model combinations that exist
    plot_data = overall_plot.dropna(subset=['accuracy'])
    
    # Only include variants that actually exist in the data
    existing_variants = plot_data['variant_label'].unique()
    filtered_hue_order = [v for v in hue_order if v in existing_variants]
    filtered_palette = [palette[hue_order.index(v)] for v in filtered_hue_order]
    
    sns.barplot(
        data=plot_data,
        x="model",
        y="accuracy",
        hue="variant_label",
        hue_order=filtered_hue_order,
        palette=filtered_palette,
        ax=ax1,
    )
    ax1.set_ylabel("Overall Accuracy (%)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Model Size", fontweight="bold", fontsize=12)
    ax1.set_title("A) Overall Accuracy Comparison", fontsize=15, fontweight="bold", loc="left")
    ax1.tick_params(axis='both', labelsize=11)
    
    # Position legend at top left
    ax1.legend(title="Configuration", fontsize=9, title_fontsize=10, 
              loc="upper center", framealpha=0.95, edgecolor='gray')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Top right: Best variant per model
    ax2 = fig.add_subplot(gs[0, 1])
    best_per_model = overall_plot.loc[overall_plot.groupby('model', observed=False)['accuracy'].idxmax()]
    model_palette = get_model_color_palette()
    colors_best = [model_palette.get(model, "#34495e") for model in best_per_model['model']]
    bars = ax2.bar(range(len(best_per_model)), best_per_model['accuracy'], color=colors_best, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(best_per_model)))
    ax2.set_xticklabels(best_per_model['model'], fontweight='semibold', rotation=15, ha='right', fontsize=11)
    ax2.set_ylabel("Best Accuracy (%)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax2.set_title("B) Best Configuration per Model", fontsize=15, fontweight="bold", loc="left")
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    # Add variant labels on bars
    for i, (bar, label) in enumerate(zip(bars, best_per_model['variant_label'])):
        height = bar.get_height()
        short_label = label.replace("GRPO: ", "").replace("Baseline (no GRPO)", "Baseline")
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, short_label,
                ha='center', va='bottom', fontsize=9, fontweight='semibold', rotation=0)
    
    # Bottom left: Difficulty trends - compare o4-mini with best Qwen model
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Find models to compare
    models_to_plot = []
    if "o4-mini (OpenAI)" in overall_plot['model'].unique():
        models_to_plot.append("o4-mini (OpenAI)")
    
    # Get the largest/best Qwen model
    qwen_models = [m for m in overall_plot['model'].unique() if m.startswith('Qwen3')]
    if qwen_models:
        # Use the largest Qwen model
        qwen_models_sorted = sorted(qwen_models, key=lambda x: float(x.replace('Qwen3-', '').replace('B', '')))
        models_to_plot.append(qwen_models_sorted[-1])
    
    # Plot difficulty curves
    for model in models_to_plot:
        model_subset = diff_plot[diff_plot['model'] == model]
        # For o4-mini, just show baseline
        if model == "o4-mini (OpenAI)":
            baseline_data = model_subset[model_subset['variant_label'] == "Baseline (no GRPO)"]
            if not baseline_data.empty:
                ax3.plot(baseline_data['difficulty'], baseline_data['accuracy'],
                        marker='o', label=f"{model}", 
                        color=get_model_color_palette().get(model, "#f39c12"),
                        linewidth=2.5, markersize=8, linestyle='--')
        else:
            # For Qwen models, show all variants
            for i, variant in enumerate(hue_order):
                subset = model_subset[model_subset['variant_label'] == variant]
                if not subset.empty:
                    ax3.plot(subset['difficulty'], subset['accuracy'], 
                            marker=['o', 's', 'D', '^', 'v'][i % 5], 
                            label=variant, color=palette[i], 
                            linewidth=2.5, markersize=8, alpha=0.9)
    
    ax3.set_xlabel("Difficulty Level", fontweight="bold", fontsize=12)
    ax3.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=12)
    title_models = " vs ".join([m.replace(" (OpenAI)", "") for m in models_to_plot])
    ax3.set_title(f"C) Difficulty Scaling: {title_models}", fontsize=15, fontweight="bold", loc="left")
    ax3.set_xticks([1, 2, 3, 4, 5])
    ax3.set_ylim(0, 60)
    ax3.tick_params(axis='both', labelsize=11)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    # Add difficulty zones (adjusted for 60 max)
    ax3.axhspan(0, 20, alpha=0.05, color='red', zorder=0)
    ax3.axhspan(20, 40, alpha=0.05, color='orange', zorder=0)
    ax3.axhspan(40, 60, alpha=0.05, color='green', zorder=0)
    
    # Bottom right: Average error rates with o4-mini as separate bar
    ax4 = fig.add_subplot(gs[1, 1])
    
    error_types_short = {
        "Invalid Paths": "Invalid\nPaths",
        "Disconnected Paths": "Disconn.\nPaths",
        "Wrong Start/End": "Wrong\nStart/End",
        "Intersecting Paths": "Intersect.\nPaths",
        "Rule Violations": "Rule\nViolations",
    }
    error_categories = list(error_types_short.values())
    x_pos = range(len(error_categories))
    
    # Separate o4-mini from other models
    o4_data = err_plot[err_plot['model'] == 'o4-mini (OpenAI)'].copy()
    qwen_data = err_plot[err_plot['model'] != 'o4-mini (OpenAI)'].copy()
    
    o4_data['error_type_short'] = o4_data['error_type'].map(error_types_short)
    qwen_data['error_type_short'] = qwen_data['error_type'].map(error_types_short)
    
    # Build custom labels and data (only include if data exists)
    labels_to_plot = []
    colors_to_plot = []
    data_to_plot = []
    
    # 1. Qwen Baseline (no GRPO)
    if not qwen_data.empty:
        qwen_baseline = qwen_data[qwen_data['variant_label'] == 'Baseline (no GRPO)']
        if not qwen_baseline.empty and len(qwen_baseline) > 0:
            vals = [qwen_baseline[qwen_baseline['error_type_short'] == et]['error_rate'].mean() 
                   if (qwen_baseline['error_type_short'] == et).any() else 0.0 
                   for et in error_categories]
            # Only add if we have some non-zero data
            if sum(vals) > 0:
                labels_to_plot.append('Baseline\n(Qwen)')
                colors_to_plot.append(palette[0])  # Same color as baseline
                data_to_plot.append(vals)
    
    # 2. o4-mini
    if not o4_data.empty and len(o4_data) > 0:
        vals = [o4_data[o4_data['error_type_short'] == et]['error_rate'].mean() 
               if (o4_data['error_type_short'] == et).any() else 0.0 
               for et in error_categories]
        # Only add if we have some non-zero data
        if sum(vals) > 0:
            labels_to_plot.append('o4-mini')
            colors_to_plot.append('#f39c12')  # Orange for o4-mini
            data_to_plot.append(vals)
    
    # 3. GRPO variants (averaged across Qwen models)
    for variant in hue_order:
        if variant == 'Baseline (no GRPO)':
            continue  # Already added
        qwen_variant = qwen_data[qwen_data['variant_label'] == variant]
        if not qwen_variant.empty and len(qwen_variant) > 0:
            vals = [qwen_variant[qwen_variant['error_type_short'] == et]['error_rate'].mean() 
                   if (qwen_variant['error_type_short'] == et).any() else 0.0 
                   for et in error_categories]
            # Only add if we have some non-zero data
            if sum(vals) > 0:
                labels_to_plot.append(variant.replace('GRPO: ', ''))
                colors_to_plot.append(palette[hue_order.index(variant)])
                data_to_plot.append(vals)
    
    # Plot bars
    width = 0.13
    for i, (label, color, vals) in enumerate(zip(labels_to_plot, colors_to_plot, data_to_plot)):
        offset = (i - len(labels_to_plot)/2) * width + width/2
        ax4.bar([x + offset for x in x_pos], vals, width, label=label, color=color)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(error_categories, fontsize=10)
    ax4.set_ylabel("Average Error Rate (%)", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Error Type", fontweight="bold", fontsize=12)
    ax4.set_title("D) Error Rate Comparison (Avg Across Models)", fontsize=15, fontweight="bold", loc="left")
    ax4.tick_params(axis='y', labelsize=11)
    ax4.legend(fontsize=9, loc="upper right", ncol=2)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)
    
    fig.suptitle("SPaRC GRPO Training Configuration Analysis - Comprehensive Summary", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "results" / "sparc"
    output_dir = base_dir / "results" / "figures"
    ensure_output_dir(output_dir)

    runs = collect_runs(input_dir)
    overall_df, diff_df, err_df = build_dataframes(runs)
    overall_df, diff_df, err_df = sort_categoricals(overall_df, diff_df, err_df)

    # Save CSVs for quick inspection
    if not overall_df.empty:
        overall_df.to_csv(output_dir / "grpo_overall_accuracy.csv", index=False)
    if not diff_df.empty:
        diff_df.to_csv(output_dir / "grpo_difficulty_accuracy.csv", index=False)
    if not err_df.empty:
        err_df.to_csv(output_dir / "grpo_error_rates.csv", index=False)

    # Individual figures
    save_overall_accuracy_plot(overall_df, output_dir / "grpo_overall_accuracy_by_variant.png")
    save_difficulty_accuracy_plot(diff_df, output_dir / "grpo_accuracy_by_difficulty.png")
    save_error_rates_plot(err_df, output_dir / "grpo_error_rates.png")
    
    # Comprehensive summary dashboard
    save_summary_dashboard(overall_df, diff_df, err_df, output_dir / "grpo_summary_dashboard.png")
    
    print("✅ All visualizations generated successfully!")
    print(f"   📊 Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()


