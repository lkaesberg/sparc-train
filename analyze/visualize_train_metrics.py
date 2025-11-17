#!/usr/bin/env python3
"""
Visualize training metrics from GRPO training runs.
Creates line charts for various reward metrics across different model sizes and variants.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# Import styling from plot_config
from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    TEXT_WIDTH_INCHES,
)


def parse_csv_file(csv_path):
    """
    Parse a training metrics CSV file and extract data for each model variant.
    Returns a dict with structure:
    {
        'epochs': [0.02, 0.04, ...],
        'models': {
            'Qwen3-32B': [value1, value2, ...],
            'Qwen3-32B-low-format': [value1, value2, ...],
            ...
        }
    }
    """
    df = pd.read_csv(csv_path)
    
    # Get the epoch column
    epochs = df['train/epoch'].values
    
    # Extract model names and their data
    models_data = {}
    
    # Define model sizes to look for
    model_sizes = ['0.6B', '4B', '14B', '32B']
    
    for size in model_sizes:
        # Look for both regular and low-format variants
        for variant_name, variant_pattern in [('', ' - '), ('-low-format', '-low-format - ')]:
            model_name = f'Qwen3-{size}{variant_name}'
            
            # Find the column with the actual metric value (not _step, not __MIN, not __MAX)
            # Use more precise pattern matching to avoid matching both regular and low-format
            metric_cols = [col for col in df.columns 
                          if f'Qwen/Qwen3-{size}{variant_pattern}' in col 
                          and not col.endswith('_step')
                          and not col.endswith('__MIN')
                          and not col.endswith('__MAX')
                          and 'train/' in col]
            
            if metric_cols:
                # Use the first matching metric column
                col = metric_cols[0]
                values = pd.to_numeric(df[col], errors='coerce').values
                
                # Only include if we have at least some valid data
                if not np.all(np.isnan(values)):
                    models_data[model_name] = values
                    print(f"  Found {model_name}: {np.sum(~np.isnan(values))}/{len(values)} valid values")
    
    return {
        'epochs': epochs,
        'models': models_data
    }


def apply_ema_smoothing(values, alpha=0.99):
    """
    Apply Exponential Moving Average smoothing to a series of values.
    
    Args:
        values: numpy array of values to smooth
        alpha: smoothing factor (0 < alpha < 1). Higher values = more smoothing.
               alpha=0.99 means 99% weight to previous EMA, 1% to current value.
    
    Returns:
        Smoothed numpy array
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]  # Start with first value
    
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            smoothed[i] = smoothed[i-1]  # Carry forward if NaN
        else:
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    
    return smoothed


def get_metric_title(filename):
    """
    Convert filename to a nice metric title.
    """
    metric_map = {
        'chart_reward.csv': 'Total Reward',
        'chart_non_intersect.csv': 'Non-Intersect Reward',
        'chart_no_rule_crossing.csv': 'No Rule Crossing Reward',
        'chart_start_end.csv': 'Start/End Reward',
        'chart_format_hint.csv': 'Format Hint Reward',
        'chart_connected_line.csv': 'Connected Line Reward',
        'chart_perfect.csv': 'Perfect Solution Reward',
        'chart_mean_terminated.csv': 'Mean Generated Tokens',
    }
    return metric_map.get(filename, filename.replace('chart_', '').replace('.csv', '').replace('_', ' ').title())


def create_line_chart(csv_path, output_dir):
    """
    Create a line chart for a single metric CSV file.
    Averages across model sizes and shows only normal vs low-format variants.
    """
    # Parse the CSV
    data = parse_csv_file(csv_path)
    epochs = data['epochs']
    models_data = data['models']
    
    if not models_data:
        print(f"No data found in {csv_path.name}")
        return
    
    # Separate normal and low-format models
    normal_models = []
    low_format_models = []
    
    for model_name, values in models_data.items():
        if 'low-format' in model_name:
            low_format_models.append(values)
            print(f"  Low-format: {model_name}")
        else:
            normal_models.append(values)
            print(f"  Standard: {model_name}")
    
    # Calculate averages and apply EMA smoothing
    averaged_data = {}
    
    if normal_models:
        # Stack and compute mean, handling NaN values
        normal_stack = np.array(normal_models)
        avg_normal = np.nanmean(normal_stack, axis=0)
        # Apply EMA smoothing
        averaged_data['Normal'] = apply_ema_smoothing(avg_normal, alpha=0.99)
        print(f"  Averaged {len(normal_models)} standard models")
    
    if low_format_models:
        low_format_stack = np.array(low_format_models)
        avg_low_format = np.nanmean(low_format_stack, axis=0)
        # Apply EMA smoothing
        averaged_data['Low-Format'] = apply_ema_smoothing(avg_low_format, alpha=0.99)
        print(f"  Averaged {len(low_format_models)} low-format models")
    
    if not averaged_data:
        print(f"No valid data to plot in {csv_path.name}")
        return
    
    print(f"  Final groups: {list(averaged_data.keys())}")
    
    # Setup plot style
    setup_plot_style(use_latex=True)
    
    # Create figure with column width
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8))
    
    # Define colors using Seaborn Set2 palette
    import matplotlib.colors as mcolors
    set2_colors = plt.cm.Set2.colors
    color_normal = mcolors.rgb2hex(set2_colors[0])      # Teal
    color_low_format = mcolors.rgb2hex(set2_colors[1])  # Orange
    
    # Plot averaged and smoothed lines (clean, no shading)
    if 'Normal' in averaged_data:
        mean = averaged_data['Normal']
        ax.plot(epochs, mean, 
                linestyle='-', 
                linewidth=2.0, 
                label='GRPO',
                color=color_normal)
    
    if 'Low-Format' in averaged_data:
        mean = averaged_data['Low-Format']
        ax.plot(epochs, mean, 
                linestyle='-', 
                linewidth=2.0, 
                label='GRPO-Low',
                color=color_low_format)
    
    # Customize plot
    ax.set_xlabel('Training Epoch')
    # Set appropriate y-axis label based on metric
    if 'mean_terminated' in csv_path.name:
        ax.set_ylabel('Tokens')
    else:
        ax.set_ylabel('Reward')
    
    # Get metric title
    metric_title = get_metric_title(csv_path.name)
    # ax.set_title(metric_title, fontsize=10, pad=10)
    
    # Legend - place it below the plot without frame (moved further down)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
              ncol=2, frameon=False, fontsize=10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set reasonable y-axis limits with padding
    all_values = np.concatenate([v[~np.isnan(v)] for v in averaged_data.values()])
    if len(all_values) > 0:
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    
    # Save figure
    output_name = csv_path.stem  # e.g., 'chart_reward'
    for ext in ['pdf', 'png', 'svg']:
        output_file = output_dir / f'{output_name}.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()


def create_combined_dashboard(train_data_dir, output_dir):
    """
    Create a combined dashboard showing all metrics in subplots.
    Averages across model sizes and shows only normal vs low-format variants.
    """
    # Get all CSV files
    csv_files = sorted(train_data_dir.glob('chart_*.csv'))
    
    if not csv_files:
        print("No chart CSV files found")
        return
    
    # Setup plot style
    setup_plot_style(use_latex=True)
    
    # Determine grid layout
    n_metrics = len(csv_files)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure with text width
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(TEXT_WIDTH_INCHES, TEXT_WIDTH_INCHES * 0.4 * n_rows))
    
    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Define colors
    import matplotlib.colors as mcolors
    set2_colors = plt.cm.Set2.colors
    color_normal = mcolors.rgb2hex(set2_colors[0])      # Teal
    color_low_format = mcolors.rgb2hex(set2_colors[1])  # Orange
    
    # Plot each metric
    for idx, csv_path in enumerate(csv_files):
        ax = axes_flat[idx]
        
        # Parse the CSV
        data = parse_csv_file(csv_path)
        epochs = data['epochs']
        models_data = data['models']
        
        if not models_data:
            ax.set_visible(False)
            continue
        
        # Separate normal and low-format models
        normal_models = []
        low_format_models = []
        
        for model_name, values in models_data.items():
            if 'low-format' in model_name:
                low_format_models.append(values)
            else:
                normal_models.append(values)
        
        # Calculate averages and apply EMA smoothing
        averaged_data = {}
        
        if normal_models:
            normal_stack = np.array(normal_models)
            avg_normal = np.nanmean(normal_stack, axis=0)
            # Apply EMA smoothing
            averaged_data['Normal'] = apply_ema_smoothing(avg_normal, alpha=0.99)
        
        if low_format_models:
            low_format_stack = np.array(low_format_models)
            avg_low_format = np.nanmean(low_format_stack, axis=0)
            # Apply EMA smoothing
            averaged_data['Low-Format'] = apply_ema_smoothing(avg_low_format, alpha=0.99)
        
        if not averaged_data:
            ax.set_visible(False)
            continue
        
        # Plot averaged lines (clean, no shading)
        if 'Normal' in averaged_data:
            mean = averaged_data['Normal']
            ax.plot(epochs, mean, 
                   linestyle='-', 
                   linewidth=1.5, 
                   label='Standard',
                   color=color_normal)
        
        if 'Low-Format' in averaged_data:
            mean = averaged_data['Low-Format']
            ax.plot(epochs, mean, 
                   linestyle='-', 
                   linewidth=1.5, 
                   label='Low Format',
                   color=color_low_format)
        
        # Customize subplot
        metric_title = get_metric_title(csv_path.name)
        ax.set_title(metric_title, fontsize=9, pad=5)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.tick_params(labelsize=7)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set reasonable y-axis limits
        all_values = np.concatenate([v[~np.isnan(v)] for v in averaged_data.values()])
        if len(all_values) > 0:
            y_min, y_max = all_values.min(), all_values.max()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # Add legend only to first subplot - below and without frame (moved further down)
        if idx == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.35, -0.35), 
                     ncol=2, frameon=False, fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save dashboard
    for ext in ['pdf', 'png', 'svg']:
        output_file = output_dir / f'train_metrics_dashboard.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()


def main():
    """
    Main function to generate all visualizations.
    """
    # Setup paths
    script_dir = Path(__file__).parent
    train_data_dir = script_dir / 'results' / 'train-data'
    output_dir = script_dir / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating training metrics visualizations...")
    print("=" * 80)
    
    # Get all CSV files
    csv_files = sorted(train_data_dir.glob('chart_*.csv'))
    
    if not csv_files:
        print(f"No chart CSV files found in {train_data_dir}")
        return
    
    print(f"Found {len(csv_files)} metric files")
    print()
    
    # Generate individual charts
    print("Generating individual metric charts...")
    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        create_line_chart(csv_path, output_dir)
    
    print("\n" + "=" * 80)
    print("Generating combined dashboard...")
    create_combined_dashboard(train_data_dir, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations complete!")


if __name__ == '__main__':
    main()

