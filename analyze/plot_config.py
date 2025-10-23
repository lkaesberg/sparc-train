# I thank Claude Sonnet 4.5 (and LBK) for its help in writing this file.
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
import os
from scipy.stats import chi2_contingency
from matplotlib.offsetbox import OffsetImage
from PIL import Image

# LaTeX document dimensions in points
# Convert to inches for matplotlib (1 pt = 1/72 inch)
TEXT_WIDTH_PT = 455.24411  # pt
COLUMN_WIDTH_PT = 219.08614  # pt
TEXT_WIDTH_INCHES = TEXT_WIDTH_PT / 72.0  # inches
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PT / 72.0  # inches


def setup_plot_style(use_latex=True):
    """
    Configure matplotlib with consistent style settings.
    """
    # Apply matplotlib style
    plt.style.use("seaborn-v0_8-muted")

    # Font configuration - Latin Modern Roman with LaTeX support
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times"]
    plt.rcParams["text.usetex"] = use_latex
    plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math

    # Size configuration
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 12

    # Line and marker configuration
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markersize"] = 4

    # Figure and export configuration
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Optional: Remove top and right spines for cleaner look
    # Commented out by default - uncomment if desired
    # plt.rcParams["axes.spines.top"] = False
    # plt.rcParams["axes.spines.right"] = False

GAMMA = "#A79D5B"
ETA = "#6C8A5B"

# University/Logo color palette
UNIBLAU = "#153268"
LOGOBLAU = "#005f9b"
LOGOHELLBLAU30 = "#d2e6fa"
LOGOHELLBLAU = "#50a5d2"
LOGOMITTELBLAU = "#0091c8"

# Categorical colormap for models
# Each model gets a unique, distinguishable color
# Colors based on Pastel6 from seaborn
MODEL_COLORS = {
    # Gemma models
    "Gemma 3 27B": "#FF615C",  #
    "Gemma 3 12B": "#FF8985",  #
    # Qwen 3 models - distinct colors for each size
    "Qwen 3 0.6B": "#E8B4FA",  # Light purple
    "Qwen 3 4B": "#C78EF0",    # Medium purple
    "Qwen 3 14B": "#A47AFF",   # Purple
    "Qwen 3 32B": "#7E4FD9",   # Dark purple
    # Qwen 2.5 models
    "Qwen 2.5 72B": "#C7ADFF",  #
    "Qwen 2.5 32B": "#C7ADFF",  #
    "Qwen 2.5 14B": "#C7ADFF",  #
    "Qwen 2.5 7B": "#C7ADFF",  #
    # Llama models
    "Llama 3.3 70B": "#69A9ED",  #
    "Llama 3.1 70B": "#91C0F2",  #
    "Llama 3.1 8B": "#A4CBF4",  #
    # DeepSeek/R1 models
    "R1 Llama Distill 70B": "#61DB7E",  #
    "Unknown 1": "#FFFC64",  #
    "Unknown 2": "#84E8E5",  #
    # Rule-based agents
    "Rule Agent": "#707070",  # Gray
    "Random Agent": "#A0A0A0",  # Light Gray
    # Human players
    "Human": "#6B5E62",  # Dark Navy
}

# Fallback color for undefined models - VERY OBVIOUS!
MODEL_COLOR_FALLBACK = "#FF00FF"  # Bright Magenta - impossible to miss!


def get_model_color(model_name, warn_on_missing=True):
    """
    Get the color for a specific model from the MODEL_COLORS dict.
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    if warn_on_missing:
        print(f"⚠️  WARNING: No color defined for model '{model_name}'!")
        print(f"   Using fallback color {MODEL_COLOR_FALLBACK} (bright magenta)")
        print(f"   Please add '{model_name}' to MODEL_COLORS in plot_config.py")

    return MODEL_COLOR_FALLBACK


def get_model_colors(model_names, warn_on_missing=True):
    """
    Get colors for multiple models.
    """
    return [get_model_color(name, warn_on_missing) for name in model_names]


def get_model_imagebox(model_name):
    """
    Get an OffsetImage (imagebox) for a model's logo.
    """
    # Internal mapping for logo files - tuples of (width, height, zoom)
    # Some logos are taller, some are wider, adjust dimensions and zoom as needed
    LOGO_CONFIG = {
        "gemma.png": (64, 64, 1/7),
        "qwen.png": (64, 64, 1/7.5),
        "deepseek.png": (64, 64, 1/6),
        "llama.png": (64, 64, 1/6),
        "human.png": (64, 64, 1/8),
    }

    LOGO_MAPPING = {
        "Human": "human.png",
        "Gemma": "gemma.png",
        "Qwen": "qwen.png",
        "R1": "deepseek.png",
        "Llama": "llama.png",
    }

    logo_path = None
    for keyword, logo in LOGO_MAPPING.items():
        if keyword in model_name:
            logo_path = Path(__file__).parent / "logos" / logo
            if logo_path.exists():
                break
    
    if not logo_path:
        return None
    
    # Load the logo with PIL
    img_pil = Image.open(str(logo_path)).convert('RGBA')
    
    # Get configuration (size and zoom) based on logo filename
    width, height, zoom = LOGO_CONFIG.get(logo_path.name)  # Default config
    
    # Resize to thumbnail size while maintaining aspect ratio
    img_pil.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    # Create and return OffsetImage with specified zoom
    imagebox = OffsetImage(np.array(img_pil), zoom=zoom)
    
    return imagebox


def perform_chi_square_test(contingency_table, test_name, group1_name, group2_name, alpha=0.05, remove_zero_columns=True, show_effect_size_interpretation=False):
    """
    Perform chi-square test for homogeneity on a contingency table.

    This is a general-purpose function for testing whether the distribution of
    categorical variables differs significantly between two or more groups.
    """
    # Convert to numpy array if needed
    if hasattr(contingency_table, "values"):  # pandas DataFrame
        data = contingency_table.copy()
        if remove_zero_columns:
            data = data.loc[:, (data != 0).any()]
        contingency_array = data.values
    else:
        contingency_array = np.array(contingency_table)
        if remove_zero_columns:
            # Remove columns that are all zeros
            contingency_array = contingency_array[:, (contingency_array != 0).any(axis=0)]

    print(f"\n--- {test_name} ---")
    if hasattr(contingency_table, "to_string"):
        print(f"Contingency table:")
        print(data.to_string() if remove_zero_columns else contingency_table.to_string())
    else:
        print(f"Contingency table shape: {contingency_array.shape}")

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

    # Calculate Cramer's V (effect size)
    n = contingency_array.sum()  # Total sample size
    min_dim = min(contingency_array.shape[0], contingency_array.shape[1]) - 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

    print(f"\nNull hypothesis: Distribution patterns are homogeneous across {group1_name} and {group2_name}")
    print(f"Alternative hypothesis: Distribution patterns differ significantly between groups")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print(f"Cramer's V (effect size): {cramers_v:.4f}")

    if show_effect_size_interpretation:
        if cramers_v < 0.1:
            effect_interpretation = "negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "small"
        elif cramers_v < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        print(f"Effect size interpretation: {effect_interpretation}")

    significant = p_value < alpha
    if significant:
        print(f"Result: SIGNIFICANT (p < {alpha}) - Distribution patterns differ significantly between groups")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha}) - No significant difference in distribution patterns")

    return {"chi2_stat": chi2_stat, "p_value": p_value, "dof": dof, "cramers_v": cramers_v, "significant": significant}


# Additional color palettes can be added here in the future
# For example:
# PLAYER_COLORS = [...]
# TECHNIQUE_COLORS = {...}
# etc.

# Helper: desaturate color for negative values
def desaturate_color(hexcolor, factor=0.4):
    """Desaturate a color by blending it with gray"""
    hexcolor = hexcolor.lstrip('#')
    r, g, b = int(hexcolor[0:2], 16), int(hexcolor[2:4], 16), int(hexcolor[4:6], 16)
    # Convert to grayscale value
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    # Blend with grayscale
    r_new = int(r * factor + gray * (1 - factor))
    g_new = int(g * factor + gray * (1 - factor))
    b_new = int(b * factor + gray * (1 - factor))
    return f'#{r_new:02x}{g_new:02x}{b_new:02x}'
