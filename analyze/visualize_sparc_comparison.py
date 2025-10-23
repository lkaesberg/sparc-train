#!/usr/bin/env python3
"""
Visualization script for SPaRC model comparison.
Creates a chart showing base model accuracy and training variant deltas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from matplotlib import gridspec
from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_color,
    desaturate_color
)


def get_overall_accuracy(stats_file):
    """
    Get overall accuracy from a _stats.csv file.
    
    Returns:
        float: Overall accuracy percentage
    """
    df = pd.read_csv(stats_file)
    
    for _, row in df.iterrows():
        if row['Metric'] == 'Correctly Solved':
            # Parse percentage string (e.g., "3.0%")
            pct_str = str(row['Percentage']).replace('%', '')
            return float(pct_str)
    
    return 0.0


def parse_model_info(filename):
    """
    Parse model size and training variant from filename.
    
    Returns:
        tuple: (size, variant) e.g., ("0.6B", "GRPO") or ("32B", "base")
    """
    # Extract size (e.g., "0.6B", "32B", "14B")
    size_match = re.search(r'(\d+\.?\d*B)', filename)
    if not size_match:
        return None, None
    
    size = size_match.group(1)
    
    # Determine if it's a base model or trained variant
    if filename.startswith('Qwen_'):
        # Base model
        return size, 'base'
    
    # Trained variants
    if '-SFT' in filename:
        return size, 'SFT'
    elif '-16R' in filename:
        return size, '16R'
    elif '-8E' in filename:
        return size, '8E'
    elif '-L' in filename:
        return size, 'L'
    elif '-GRPO' in filename:
        return size, 'GRPO'
    
    return size, 'unknown'


def collect_model_data(results_dir):
    """
    Collect all model data organized by size and variant.
    
    Returns:
        dict: {size: {'base': accuracy, 'GRPO': accuracy, ...}}
    """
    data = {}
    
    # Get all stats files
    stats_files = list(results_dir.glob('*_stats.csv'))
    
    for stats_file in stats_files:
        size, variant = parse_model_info(stats_file.name)
        if size is None:
            continue
        
        accuracy = get_overall_accuracy(stats_file)
        
        if size not in data:
            data[size] = {}
        
        data[size][variant] = accuracy
    
    return data


def create_comparison_chart(results_dir, output_path=None, model_sizes=None):
    """
    Create a comparison chart matching the image layout.
    
    Args:
        results_dir: Path to directory containing _stats.csv files
        output_path: Where to save the figure (optional)
        model_sizes: List of model sizes to include (e.g., ['0.6B', '4B', '14B', '32B'])
    """
    setup_plot_style(use_latex=True)
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.labelsize"] = 8

    
    # Collect all model data
    all_data = collect_model_data(results_dir)
    
    # Determine which sizes to plot
    if model_sizes is None:
        model_sizes = sorted(all_data.keys(), key=lambda x: float(x.replace('B', '')))
    
    # Filter to only models we want
    model_sizes = [s for s in model_sizes if s in all_data]
    
    if not model_sizes:
        print("No model data found!")
        return
    
    n_models = len(model_sizes)
    
    # Define training variants (rows in the bottom section)
    # Order requested: L, 16R, 8E, GRPO, SFT (reversed)
    variants = ['16R', '8E', 'L', 'GRPO', 'SFT']
    variant_labels = {v: v for v in variants}
    
    # Calculate figure dimensions
    fig_width = TEXT_WIDTH_INCHES
    # Top plot has 1 bar, bottom plot has 5 bars - use height ratio 1:5
    # Compact layout: less spacing, tighter bars
    row_height = 0.35  # inches per bar row (reduced from 0.5)
    fig_height = 6 * row_height + 0.6  # 6 total bar rows + minimal spacing
    print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches")
    
    # Create figure with height ratios (1 for top, 5 for bottom)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, n_models, height_ratios=[1, 5], hspace=0.5, wspace=0.15)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_models)] for i in range(2)])
    
    # Adjust spacing (already set in gridspec)
    # plt.subplots_adjust(hspace=0.5, wspace=0.15)
    
    # Ensure axes is 2D array even for single column
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    # Top row: Base model accuracy (Full Set)
    for col_idx, size in enumerate(model_sizes):
        ax = axes[0, col_idx]
        
        # Get base model accuracy and color
        base_acc = all_data[size].get('base', 0.0)
        model_name = f"Qwen 3 {size}"
        color = get_model_color(model_name)

        # Create horizontal bar (single-row) and label
        bar = ax.barh([0], [base_acc], color=color, height=0.6)

        # Position label on the right side of bar with black text
        ax.text(base_acc + 0.5, 0, f'{base_acc:.2f}', ha='left', va='center', fontweight='bold', color='black', fontsize=8)

        # Configure axis
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        
        if col_idx == 0:
            ax.set_yticklabels(['Baseline'])
        else:
            ax.set_yticklabels([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Accuracy (\\%)')
        ax.set_title(model_name)
    
    # Bottom row: Training variant deltas (Subset)
    for col_idx, size in enumerate(model_sizes):
        ax = axes[1, col_idx]
        
        base_acc = all_data[size].get('base', 0.0)
        model_name = f"Qwen 3 {size}"
        color = get_model_color(model_name)

        # Calculate deltas for each variant
        deltas = []
        for variant in variants:
            variant_acc = all_data[size].get(variant, None)
            if variant_acc is not None:
                delta = variant_acc - base_acc
            else:
                delta = np.nan
            deltas.append(delta)

        # Plot bars (use zero baseline and show delta values)
        y_pos = np.arange(len(variants))
        bar_colors = [color if (not np.isnan(d) and d >= 0) else desaturate_color(color, 0.3) for d in deltas]
        bars = ax.barh(y_pos, [0 if np.isnan(d) else d for d in deltas], color=bar_colors, height=0.6)

        # Add value labels (skip NaNs) - place on right side with black text
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            if np.isnan(delta) or abs(delta) < 0.1:
                continue
            width = bar.get_width()
            
            # Position label on the right for positive, left for negative
            if delta >= 0:
                label_x = width + 0.5
                ha = 'left'
            else:
                label_x = width - 0.5
                ha = 'right'
            
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{delta:.1f}', 
                   ha=ha, va='center', color='black', fontweight='bold', fontsize=8)
        
        # Configure axis
        ax.set_xlim(-19, 19)
        ax.set_ylim(-0.5, len(variants) - 0.5)
        ax.set_yticks(y_pos)
        
        if col_idx == 0:
            ax.set_yticklabels([variant_labels[v] for v in variants])
            ax.set_ylabel('Training Method', fontweight='bold')
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel('$\\Delta$ Accuracy (\\%)')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
        
        # Add separator line after SFT (between SFT and GRPO-based methods)
        # SFT is at index 4 in our reversed list, so separator goes at y=3.5
        ax.axhline(3.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    
    return fig


def main():
    """
    Main function to generate the comparison chart.
    """
    # Define the results directory
    results_dir = Path(__file__).parent / 'results' / 'sparc'
    
    # Specify which model sizes to include (in order)
    model_sizes = ['0.6B', '4B', '14B', '32B']
    
    # Create output directory
    output_dir = results_dir.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'model_comparison.pdf'
    
    # Generate the chart
    fig = create_comparison_chart(results_dir, output_path, model_sizes=model_sizes)
    
    if fig:
        plt.show()
    else:
        print("Failed to generate chart.")


if __name__ == '__main__':
    main()
