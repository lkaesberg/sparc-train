#!/usr/bin/env python3
"""
Visualization script for annotation error comparison with delta charts.
Creates a chart showing baseline error rates and training variant deltas.

Similar to visualize_sparc_comparison.py but for annotation errors instead of accuracy.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from matplotlib import gridspec
from plot_config import (
    setup_plot_style,
    TEXT_WIDTH_INCHES,
    COLUMN_WIDTH_INCHES,
    get_model_color,
    desaturate_color
)


def read_annotation_file(filepath):
    """Read a JSONL annotation file and extract error category counts."""
    error_counts = defaultdict(int)
    total_samples = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            total_samples += 1
            
            # Extract error categories from llm_annotation
            if 'llm_annotation' in sample and 'categories' in sample['llm_annotation']:
                categories = sample['llm_annotation']['categories']
                for cat in categories:
                    error_counts[cat] += 1
    
    return error_counts, total_samples


def calculate_error_rates(error_counts, total_samples):
    """Convert error counts to error rates (percentage)."""
    error_types = ['A', 'B', 'C', 'D', 'E', 'F']
    error_rates = {}
    
    for error_type in error_types:
        count = error_counts.get(error_type, 0)
        rate = (count / total_samples * 100) if total_samples > 0 else 0
        error_rates[error_type] = rate
    
    return error_rates


def get_config_from_filename(filename):
    """Determine configuration type from filename."""
    stem = filename.stem
    
    # Extract model size first (needed for all configs)
    size_match = None
    for size in ['0.6B', '4B', '14B', '32B']:
        # For baseline: Qwen_Qwen3-4B
        # For trained: lkaesberg_Qwen3-4B-SPaRC-...
        if f'-{size}' in stem or f'{size}' in stem:
            size_match = size
            break
    
    # Baseline: Qwen_Qwen3-*
    if stem.startswith('Qwen_Qwen3-'):
        return 'Baseline', size_match
    
    # SFT
    if 'SPaRC-SFT' in stem:
        return 'SFT', size_match
    
    # GRPO variants
    if 'SPaRC-GRPO-L' in stem:
        return 'GRPO-L', size_match
    elif 'SPaRC-GRPO-8E' in stem:
        return 'GRPO-8E', size_match
    elif 'SPaRC-GRPO-16R' in stem:
        return 'GRPO-16R', size_match
    elif 'SPaRC-GRPO' in stem:
        return 'GRPO', size_match
    
    return None, None


def collect_annotation_data(results_dir):
    """
    Collect all annotation data organized by configuration and error type.
    
    Returns:
        dict: {config: {error_type: [rates across model sizes]}}
    """
    # Collect raw data per config per size
    data_by_config_size = defaultdict(lambda: defaultdict(dict))
    
    # Process all annotation files
    for filepath in results_dir.glob('*.annotated_by_openai_gpt-oss-120b.jsonl'):
        config_name, model_size = get_config_from_filename(filepath)
        
        if config_name is None:
            print(f"Skipping {filepath.name}: could not determine config")
            continue
        
        if model_size is None:
            print(f"Skipping {filepath.name}: could not determine model size")
            continue
        
        # Read error counts
        error_counts, total_samples = read_annotation_file(filepath)
        
        if total_samples == 0:
            print(f"Skipping {filepath.name}: no samples found")
            continue
        
        # Calculate error rates
        error_rates = calculate_error_rates(error_counts, total_samples)
        
        # Store by config and size
        data_by_config_size[config_name][model_size] = error_rates
        print(f"Loaded {config_name} ({model_size}): {total_samples} samples")
    
    # Average across model sizes for each config
    averaged_data = {}
    for config_name, size_data in data_by_config_size.items():
        config_rates = defaultdict(list)
        for model_size, error_rates in size_data.items():
            for error_type, rate in error_rates.items():
                config_rates[error_type].append(rate)
        
        # Average each error type across sizes
        averaged_data[config_name] = {
            error_type: np.mean(rates) 
            for error_type, rates in config_rates.items()
        }
        print(f"Averaged {config_name}: {len(size_data)} model sizes")
    
    return averaged_data


def create_delta_chart(results_dir, output_path=None, exclude_categories=None, 
                      exclude_configs=None, include_configs=None):
    """
    Create a delta comparison chart for annotation errors.
    
    Args:
        results_dir: Path to directory containing annotation files
        output_path: Where to save the figure (optional)
        exclude_categories: List of error categories to exclude
        exclude_configs: List of configurations to exclude
        include_configs: List of configurations to include (if set, only these are shown)
    """
    setup_plot_style(use_latex=True)
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"
    
    # Collect all annotation data
    all_data = collect_annotation_data(results_dir)
    
    if 'Baseline' not in all_data:
        print("No baseline data found!")
        return None
    
    # Define error types with display labels
    all_error_types = ['A', 'B', 'C', 'D', 'E', 'F']
    all_error_labels = [
        'Planning/\nLogical',
        'Misunderstood\nRule',
        'Spatial/\nGeometric',
        'Premature\nVerification',
        'No\nCorrection',
        'Coordinate\nError'
    ]
    
    # Apply category filtering
    if exclude_categories:
        exclude_categories = [cat.upper() for cat in exclude_categories]
        error_types = [et for et in all_error_types if et not in exclude_categories]
        error_labels = [all_error_labels[i] for i, et in enumerate(all_error_types) 
                       if et not in exclude_categories]
        print(f"Excluding categories: {', '.join(exclude_categories)}")
    else:
        error_types = all_error_types
        error_labels = all_error_labels
    
    # Define training variants (rows in the bottom section)
    # Reversed order so they appear bottom-to-top in the chart
    all_configs = ['GRPO-16R', 'GRPO-8E', 'GRPO-L', 'GRPO', 'SFT']
    
    # Apply configuration filtering
    if include_configs:
        variants = [c for c in all_configs if c in include_configs and c in all_data]
        print(f"Including only configurations: {', '.join(include_configs)}")
    elif exclude_configs:
        variants = [c for c in all_configs if c not in exclude_configs and c in all_data]
        print(f"Excluding configurations: {', '.join(exclude_configs)}")
    else:
        variants = [c for c in all_configs if c in all_data]
    
    if not variants:
        print("No training variants found after filtering!")
        return None
    
    # Variant labels with subscripts
    variant_labels = {
        'SFT': '$\\mathrm{Base}_{\\scriptscriptstyle\\color{gray}\\mathrm{SFT}}$',
        'GRPO': '$\\mathrm{Base}_{\\scriptscriptstyle\\color{gray}\\mathrm{GRPO}}$',
        'GRPO-8E': '$\\mathrm{8Ep}_{\\scriptscriptstyle\\color{gray}\\mathrm{GRPO}}$',
        'GRPO-16R': '$\\mathrm{16R}_{\\scriptscriptstyle\\color{gray}\\mathrm{GRPO}}$',
        'GRPO-L': '$\\mathrm{Low}_{\\scriptscriptstyle\\color{gray}\\mathrm{GRPO}}$',
    }
    
    n_categories = len(error_types)
    
    # Calculate figure dimensions
    fig_width = TEXT_WIDTH_INCHES
    row_height = 0.35  # inches per bar row
    fig_height = (1 + len(variants)) * row_height + 0.6  # 1 baseline + N variants + spacing
    print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches")
    
    # Create figure with height ratios (1 for baseline, N for variants)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, n_categories, height_ratios=[1, len(variants)], 
                         hspace=0.5, wspace=0.15)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_categories)] for i in range(2)])
    
    # Ensure axes is 2D array even for single column
    if n_categories == 1:
        axes = axes.reshape(2, 1)
    
    # Define colors for error categories using seaborn Set2 palette
    set2_colors = sns.color_palette("Set2", 8)
    category_colors = {
        'A': set2_colors[0],  # Light teal
        'B': set2_colors[1],  # Orange
        'C': set2_colors[2],  # Light green
        'D': set2_colors[3],  # Pink/Red
        'E': set2_colors[4],  # Purple
        'F': set2_colors[5],  # Yellow/Tan
    }
    
    # Get baseline error rates
    baseline_data = all_data['Baseline']
    
    # Top row: Baseline error rates
    for col_idx, error_type in enumerate(error_types):
        ax = axes[0, col_idx]
        
        # Get baseline rate and color
        baseline_rate = baseline_data.get(error_type, 0.0)
        color = category_colors.get(error_type, '#95A5A6')
        
        # Create horizontal bar (single-row) and label
        bar = ax.barh([0], [baseline_rate], color=color, height=0.6)
        
        # Position label on the right side of bar with black text
        # Adjust position to avoid overlap
        label_x = baseline_rate + 2
        ax.text(label_x, 0, f'{baseline_rate:.1f}', 
               ha='left', va='center', fontweight='bold', color='black', fontsize=8)
        
        # Configure axis with fixed scale
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        
        if col_idx == 0:
            ax.set_yticklabels(['Baseline'])
        else:
            ax.set_yticklabels([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Error Rate (\\%)')
        ax.set_title(error_labels[col_idx])
    
    # Bottom row: Training variant deltas
    for col_idx, error_type in enumerate(error_types):
        ax = axes[1, col_idx]
        
        baseline_rate = baseline_data.get(error_type, 0.0)
        color = category_colors.get(error_type, '#95A5A6')
        
        # Calculate deltas for each variant
        deltas = []
        for variant in variants:
            variant_rate = all_data[variant].get(error_type, None)
            if variant_rate is not None:
                delta = variant_rate - baseline_rate
            else:
                delta = np.nan
            deltas.append(delta)
        
        # Plot bars (use zero baseline and show delta values)
        y_pos = np.arange(len(variants))
        bar_colors = [color if not np.isnan(d) else desaturate_color(color, 0.3) for d in deltas]
        bars = ax.barh(y_pos, [0 if np.isnan(d) else d for d in deltas], 
                      color=bar_colors, height=0.6)
        
        # Add value labels (skip NaNs)
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            if np.isnan(delta) or abs(delta) < 0.1:
                continue
            width = bar.get_width()
            
            # Position label on the right for positive, left for negative
            # Increase offset to avoid overlap
            if delta >= 0:
                label_x = width + 2
                ha = 'left'
            else:
                label_x = width - 2
                ha = 'right'
            
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{delta:.1f}', 
                   ha=ha, va='center', color='black', fontweight='bold', fontsize=8)
        
        # Configure axis with fixed scale (-60 to +60)
        ax.set_xlim(-80, 80)
        ax.set_ylim(-0.5, len(variants) - 0.5)
        ax.set_yticks(y_pos)
        
        if col_idx == 0:
            ax.set_yticklabels([variant_labels.get(v, v) for v in variants])
            ax.set_ylabel('Training Method', fontweight='bold')
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel('$\\Delta$ Error Rate (\\%)')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
        
        # Add separator line between SFT and GRPO variants
        if 'SFT' in variants and any('GRPO' in v for v in variants):
            sft_idx = variants.index('SFT')
            ax.axhline(sft_idx + 0.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Delta chart saved to: {output_path}")
    
    return fig


def main():
    """Main function to generate the delta comparison chart."""
    parser = argparse.ArgumentParser(
        description='Generate delta comparison chart for annotation errors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate chart with all categories and configurations
  python visualize_annotation_deltas.py
  
  # Exclude specific error categories
  python visualize_annotation_deltas.py --exclude-categories A D
  
  # Exclude specific configurations
  python visualize_annotation_deltas.py --exclude-configs GRPO-16R
  
  # Show only specific configurations
  python visualize_annotation_deltas.py --include-configs SFT GRPO
  
Available categories: A, B, C, D, E, F
Available configurations: SFT, GRPO, GRPO-8E, GRPO-16R, GRPO-L
        """
    )
    
    parser.add_argument('--exclude-categories', nargs='+', metavar='CAT',
                       help='Error categories to exclude (e.g., A D)')
    parser.add_argument('--exclude-configs', nargs='+', metavar='CONFIG',
                       help='Configurations to exclude (e.g., GRPO-16R)')
    parser.add_argument('--include-configs', nargs='+', metavar='CONFIG',
                       help='Configurations to include (if set, only these are shown)')
    parser.add_argument('--results-dir', type=Path,
                       default=Path(__file__).parent / 'results' / 'annotate',
                       help='Directory containing annotation files')
    parser.add_argument('--output-dir', type=Path,
                       default=Path(__file__).parent / 'results' / 'figures',
                       help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.include_configs and args.exclude_configs:
        parser.error("Cannot use both --include-configs and --exclude-configs")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename suffix based on filters
    suffix = ''
    if args.exclude_categories:
        suffix += f"_excl_cat_{''.join(args.exclude_categories)}"
    if args.exclude_configs:
        suffix += f"_excl_cfg_{'_'.join(c.replace('-', '') for c in args.exclude_configs)}"
    if args.include_configs:
        suffix += f"_incl_cfg_{'_'.join(c.replace('-', '') for c in args.include_configs)}"
    
    output_path = args.output_dir / f'annotation_error_deltas{suffix}.pdf'
    
    print(f"\n{'='*60}")
    print(f"Generating annotation error delta chart")
    print(f"{'='*60}")
    
    # Generate the chart
    fig = create_delta_chart(
        args.results_dir,
        output_path,
        exclude_categories=args.exclude_categories,
        exclude_configs=args.exclude_configs,
        include_configs=args.include_configs
    )
    
    if fig:
        print(f"\n{'='*60}")
        print(f"Delta chart generated successfully!")
        print(f"{'='*60}\n")
    else:
        print("Failed to generate delta chart.")


if __name__ == '__main__':
    main()

