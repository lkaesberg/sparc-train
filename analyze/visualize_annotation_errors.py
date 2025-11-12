"""
Generate radar plot showing annotation error rates averaged over all model sizes.
Compares Baseline, SFT, and GRPO configurations based on LLM annotations.

Error categories:
- A — Planning / Logical Reasoning Flaw
- B — Misunderstood or Invented Rule
- C — Spatial / Geometric Misjudgment
- D — Premature Verification / Overconfidence
- E — No Correction Despite Noticing Issue
- F — Grid / Coordinate Error

Usage:
    python visualize_annotation_errors.py
    python visualize_annotation_errors.py --exclude-categories A D
    python visualize_annotation_errors.py --exclude-configs Baseline GRPO-16R
    python visualize_annotation_errors.py --include-configs Baseline SFT GRPO
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
from collections import defaultdict
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES

# Setup plot style
setup_plot_style(use_latex=True)


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
    
    # Baseline: Qwen_Qwen3-*
    if stem.startswith('Qwen_Qwen3-'):
        return 'Baseline', None
    
    # Extract model size
    size_match = None
    for size in ['0.6B', '4B', '14B', '32B']:
        if f'-{size}-' in stem:
            size_match = size
            break
    
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


def generate_averaged_config_plot(results_dir, output_dir, exclude_categories=None, 
                                 exclude_configs=None, include_configs=None):
    """
    Generate a radar plot averaging over all model sizes to show configuration differences.
    Compares Baseline, SFT, and GRPO configurations showing error rates from annotations.
    
    Args:
        results_dir: Directory containing annotation files
        output_dir: Directory to save output plots
        exclude_categories: List of error categories to exclude (e.g., ['A', 'D'])
        exclude_configs: List of configurations to exclude (e.g., ['Baseline', 'GRPO-16R'])
        include_configs: List of configurations to include (if set, only these are shown)
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all error types with display labels
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
    
    # Define colors for configurations (matching the original script)
    config_colors = {
        'Baseline': '#B0B0B0',  # Gray for baseline
        'SFT': '#E8B4FA',       # Light purple
        'GRPO': '#C78EF0',      # Medium purple
        'GRPO-8E': '#A47AFF',   # Purple
        'GRPO-16R': '#8B5FE8',  # Dark purple
        'GRPO-L': '#7E4FD9',    # Darkest purple
    }
    
    config_markers = {
        'Baseline': 'X',     # X marker
        'SFT': 'o',          # Circle
        'GRPO': 's',         # Square
        'GRPO-8E': '^',      # Triangle up
        'GRPO-16R': 'D',     # Diamond
        'GRPO-L': 'P',       # Plus (filled)
    }
    
    # Collect data for each configuration
    config_data = defaultdict(list)
    
    # Process all annotation files
    for filepath in results_dir.glob('*.annotated_by_openai_gpt-oss-120b.jsonl'):
        config_name, model_size = get_config_from_filename(filepath)
        
        if config_name is None:
            continue
        
        # Read error counts
        error_counts, total_samples = read_annotation_file(filepath)
        
        if total_samples == 0:
            continue
        
        # Calculate error rates
        error_rates = calculate_error_rates(error_counts, total_samples)
        
        # Store rates in order of ALL error_types (before filtering)
        rates = [error_rates.get(et, 0) for et in all_error_types]
        config_data[config_name].append(rates)
        
        print(f"Processed {filepath.name}: {config_name}, Size: {model_size}, Samples: {total_samples}")
    
    # Average across all model sizes for each configuration
    config_averaged_data = {}
    for config_name, all_rates in config_data.items():
        if all_rates:
            # Average all rates first (based on full error type list)
            averaged = np.mean(all_rates, axis=0)
            # Then filter to only include non-excluded categories
            if exclude_categories:
                filtered_rates = [averaged[i] for i, et in enumerate(all_error_types) 
                                 if et not in exclude_categories]
                config_averaged_data[config_name] = np.array(filtered_rates)
            else:
                config_averaged_data[config_name] = averaged
            print(f"{config_name}: averaged {len(all_rates)} models")
    
    # Apply configuration filtering
    all_configs = ['Baseline', 'SFT', 'GRPO', 'GRPO-8E', 'GRPO-16R', 'GRPO-L']
    
    if include_configs:
        # Only include specified configs
        plot_order = [c for c in all_configs if c in include_configs and c in config_averaged_data]
        print(f"Including only configurations: {', '.join(include_configs)}")
    elif exclude_configs:
        # Exclude specified configs
        plot_order = [c for c in all_configs if c not in exclude_configs and c in config_averaged_data]
        print(f"Excluding configurations: {', '.join(exclude_configs)}")
    else:
        # Use all available configs
        plot_order = [c for c in all_configs if c in config_averaged_data]
    
    if not config_averaged_data or not plot_order:
        print("No data found for averaged configuration plot after filtering")
        return
    
    # Create radar plot
    fig = plt.figure(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('white')
    
    N = len(error_types)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta = np.append(theta, theta[0])
    
    # Determine y-axis limit based on max error rate in filtered data
    visible_rates = [config_averaged_data[c] for c in plot_order]
    max_rate = max(max(rates) for rates in visible_rates)
    y_max = min(100, ((max_rate // 10) + 2) * 10)  # Round up to nearest 10, cap at 100
    ax.set_ylim(0, y_max)
    
    # Plot configurations in the determined order
    for config_name in plot_order:
        
        rates_closed = np.append(config_averaged_data[config_name], 
                                config_averaged_data[config_name][0])
        
        color = config_colors[config_name]
        marker = config_markers[config_name]
        
        ax.plot(theta, rates_closed, marker=marker, linestyle='-', 
               linewidth=1.5, color=color, 
               label=config_name, alpha=0.9, markersize=4,
               markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.0,
               zorder=1)
        
        ax.fill(theta, rates_closed, alpha=0.1, color=color, zorder=1)
    
    # Set x-axis labels (error categories)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(error_labels)
    ax.tick_params(axis='x', pad=8)
    
    # Set y-axis ticks
    y_ticks = np.arange(20, y_max + 1, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}\\%' for y in y_ticks])
    
    # Grid styling
    ax.yaxis.grid(True, color='#888888', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.xaxis.grid(True, color='#888888', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.spines['polar'].set_visible(False)
    
    # Set y-axis labels to appear in front of plot lines with white outline
    ax.tick_params(axis='y', which='major', pad=-20, labelsize=9)
    for label in ax.yaxis.get_ticklabels():
        label.set_zorder(100)
        label.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal()
        ])
    
    # Legend
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                     ncol=3, frameon=False, facecolor='white',
                     edgecolor='#dddddd', framealpha=1.0,
                     handletextpad=0.5, columnspacing=1.0)
    
    plt.tight_layout()
    
    # Save plot with appropriate filename based on filters
    suffix = ''
    if exclude_categories:
        suffix += f"_excl_cat_{''.join(exclude_categories)}"
    if exclude_configs:
        suffix += f"_excl_cfg_{'_'.join(c.replace('-', '') for c in exclude_configs)}"
    if include_configs:
        suffix += f"_incl_cfg_{'_'.join(c.replace('-', '') for c in include_configs)}"
    
    output_base = output_dir / f'radar_annotation_errors_averaged_configs{suffix}'
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"Averaged configuration plot saved: {output_base}.pdf and {output_base}.png")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate radar plot of annotation errors across configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plot with all categories and configurations
  python visualize_annotation_errors.py
  
  # Exclude specific error categories
  python visualize_annotation_errors.py --exclude-categories A D
  
  # Exclude specific configurations
  python visualize_annotation_errors.py --exclude-configs Baseline GRPO-16R
  
  # Show only specific configurations
  python visualize_annotation_errors.py --include-configs Baseline SFT GRPO
  
  # Combine filters
  python visualize_annotation_errors.py --exclude-categories A --exclude-configs GRPO-16R

Available categories: A, B, C, D, E, F
Available configurations: Baseline, SFT, GRPO, GRPO-8E, GRPO-16R, GRPO-L
        """
    )
    
    parser.add_argument('--exclude-categories', nargs='+', metavar='CAT',
                       help='Error categories to exclude (e.g., A D)')
    parser.add_argument('--exclude-configs', nargs='+', metavar='CONFIG',
                       help='Configurations to exclude (e.g., Baseline GRPO-16R)')
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
    
    print(f"\n{'='*60}")
    print(f"Generating averaged configuration radar plot (annotation errors)")
    print(f"{'='*60}")
    
    generate_averaged_config_plot(
        args.results_dir, 
        args.output_dir,
        exclude_categories=args.exclude_categories,
        exclude_configs=args.exclude_configs,
        include_configs=args.include_configs
    )
    
    print(f"\n{'='*60}")
    print(f"Radar plot generated successfully!")
    print(f"{'='*60}\n")

