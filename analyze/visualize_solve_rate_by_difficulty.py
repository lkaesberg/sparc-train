#!/usr/bin/env python3
"""
Visualize solve rates by difficulty for different training configurations.
Compares baseline (base Qwen models averaged), SFT, GRPO-L, and step-by-step approaches.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np

# Import styling from plot_config
from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    MODEL_COLORS,
    get_model_color,
    get_training_method_color,
)


def parse_sparc_stats(file_path):
    """
    Parse a SPaRC stats CSV file and extract solve rates by difficulty.
    Returns a dict: {1: 26.7, 2: 9.3, ...} (percentages)
    """
    solve_rates = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Look for lines like: "Difficulty 1 Solved,23/86,26.7%"
            match = re.match(r'Difficulty (\d+) Solved,\d+/\d+,([\d.]+)%', line)
            if match:
                difficulty = int(match.group(1))
                percentage = float(match.group(2))
                solve_rates[difficulty] = percentage
    
    return solve_rates


def parse_step_by_step_summary(file_path):
    """
    Parse a step-by-step summary file and extract solve rates by difficulty.
    Returns a dict: {1: 23.26, 2: 12.71, ...} (percentages)
    """
    solve_rates = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Look for patterns like: "Difficulty 1\n  total puzzles:        86\n  wins:       23.26% (20)"
    pattern = r'Difficulty (\d+).*?wins:\s+([\d.]+)%'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        difficulty = int(match[0])
        percentage = float(match[1])
        solve_rates[difficulty] = percentage
    
    return solve_rates


def get_baseline_solve_rates(sparc_dir):
    """
    Calculate average solve rates across base Qwen models.
    """
    base_models = [
        'Qwen_Qwen3-0.6B_stats.csv',
        'Qwen_Qwen3-4B_stats.csv',
        'Qwen_Qwen3-14B_stats.csv',
        'Qwen_Qwen3-32B_stats.csv',
    ]
    
    all_rates = []
    for model_file in base_models:
        file_path = sparc_dir / model_file
        if file_path.exists():
            rates = parse_sparc_stats(file_path)
            all_rates.append(rates)
    
    # Average across models
    baseline_rates = {}
    for difficulty in [1, 2, 3, 4, 5]:
        rates_for_diff = [r[difficulty] for r in all_rates if difficulty in r]
        if rates_for_diff:
            baseline_rates[difficulty] = np.mean(rates_for_diff)
    
    return baseline_rates


def get_config_solve_rates(sparc_dir, pattern, model_sizes=['0.6B', '4B', '14B', '32B']):
    """
    Get solve rates for a specific configuration (SFT or GRPO-L) averaged over model sizes.
    """
    all_rates = []
    
    for size in model_sizes:
        # Try to find matching file
        search_pattern = f'lkaesberg_Qwen3-{size}-SPaRC-{pattern}_stats.csv'
        file_path = sparc_dir / search_pattern
        
        if file_path.exists():
            rates = parse_sparc_stats(file_path)
            all_rates.append(rates)
    
    # Average across model sizes
    config_rates = {}
    for difficulty in [1, 2, 3, 4, 5]:
        rates_for_diff = [r[difficulty] for r in all_rates if difficulty in r]
        if rates_for_diff:
            config_rates[difficulty] = np.mean(rates_for_diff)
    
    return config_rates


def get_config_deltas(sparc_dir, pattern, model_sizes=['0.6B', '4B', '14B', '32B']):
    """
    Calculate deltas for each model size, then average the deltas.
    This is more accurate than averaging rates then computing delta.
    """
    all_deltas = []
    
    for size in model_sizes:
        # Get baseline for this size
        baseline_file = sparc_dir / f'Qwen_Qwen3-{size}_stats.csv'
        # Get config for this size
        config_file = sparc_dir / f'lkaesberg_Qwen3-{size}-SPaRC-{pattern}_stats.csv'
        
        if baseline_file.exists() and config_file.exists():
            baseline_rates = parse_sparc_stats(baseline_file)
            config_rates = parse_sparc_stats(config_file)
            
            # Calculate delta for this model size
            size_deltas = {}
            for difficulty in [1, 2, 3, 4, 5]:
                if difficulty in baseline_rates and difficulty in config_rates:
                    size_deltas[difficulty] = config_rates[difficulty] - baseline_rates[difficulty]
            
            all_deltas.append(size_deltas)
    
    # Average deltas across model sizes
    avg_deltas = {}
    for difficulty in [1, 2, 3, 4, 5]:
        deltas_for_diff = [d[difficulty] for d in all_deltas if difficulty in d]
        if deltas_for_diff:
            avg_deltas[difficulty] = np.mean(deltas_for_diff)
    
    return avg_deltas


def get_step_by_step_deltas(sparc_dir, sbs_dir, model_sizes=['0.6B', '4B', '14B', '32B']):
    """
    Calculate step-by-step deltas for each model size, then average.
    """
    all_deltas = []
    
    for size in model_sizes:
        # Get baseline for this size
        baseline_file = sparc_dir / f'Qwen_Qwen3-{size}_stats.csv'
        # Get step-by-step for this size
        sbs_file = sbs_dir / f'summary_by_difficulty_Qwen3-{size}.txt'
        
        if baseline_file.exists() and sbs_file.exists():
            baseline_rates = parse_sparc_stats(baseline_file)
            sbs_rates = parse_step_by_step_summary(sbs_file)
            
            # Calculate delta for this model size
            size_deltas = {}
            for difficulty in [1, 2, 3, 4, 5]:
                if difficulty in baseline_rates and difficulty in sbs_rates:
                    size_deltas[difficulty] = sbs_rates[difficulty] - baseline_rates[difficulty]
            
            all_deltas.append(size_deltas)
    
    # Average deltas across model sizes
    avg_deltas = {}
    for difficulty in [1, 2, 3, 4, 5]:
        deltas_for_diff = [d[difficulty] for d in all_deltas if difficulty in d]
        if deltas_for_diff:
            avg_deltas[difficulty] = np.mean(deltas_for_diff)
    
    return avg_deltas


def get_step_by_step_rates(sbs_dir, model_sizes=['0.6B', '4B', '14B', '32B']):
    """
    Get solve rates for step-by-step approach averaged over model sizes.
    """
    all_rates = []
    
    for size in model_sizes:
        file_path = sbs_dir / f'summary_by_difficulty_Qwen3-{size}.txt'
        
        if file_path.exists():
            rates = parse_step_by_step_summary(file_path)
            all_rates.append(rates)
    
    # Average across model sizes
    sbs_rates = {}
    for difficulty in [1, 2, 3, 4, 5]:
        rates_for_diff = [r[difficulty] for r in all_rates if difficulty in r]
        if rates_for_diff:
            sbs_rates[difficulty] = np.mean(rates_for_diff)
    
    return sbs_rates


def create_visualization(output_dir):
    """
    Create the main visualization showing solve rates by difficulty.
    """
    # Setup paths
    sparc_dir = Path(__file__).parent / 'results' / 'sparc'
    sbs_dir = Path(__file__).parent / 'results' / 'step-by-step'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data for all configurations
    print("Loading baseline (base Qwen models)...")
    baseline_rates = get_baseline_solve_rates(sparc_dir)
    
    print("Calculating SFT deltas (per model size, then averaged)...")
    sft_deltas_dict = get_config_deltas(sparc_dir, 'SFT')
    
    print("Calculating GRPO-L deltas (per model size, then averaged)...")
    grpo_l_deltas_dict = get_config_deltas(sparc_dir, 'GRPO-L')
    
    print("Calculating step-by-step deltas (per model size, then averaged)...")
    sbs_deltas_dict = get_step_by_step_deltas(sparc_dir, sbs_dir)
    
    # Setup plot style
    setup_plot_style(use_latex=True)
    
    # Create figure with column width
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8))
    
    # Prepare data for plotting
    difficulties = [1, 2, 3, 4, 5]
    
    # Extract baseline values (for reference in stats)
    baseline_values = [baseline_rates.get(d, 0) for d in difficulties]
    
    # Extract deltas (already computed per model size, then averaged)
    sft_delta = [sft_deltas_dict.get(d, 0) for d in difficulties]
    grpo_l_delta = [grpo_l_deltas_dict.get(d, 0) for d in difficulties]
    sbs_delta = [sbs_deltas_dict.get(d, 0) for d in difficulties]
    
    # Create line plots with markers for deltas using consistent colors from plot_config
    ax.plot(difficulties, sft_delta, marker='s', linewidth=1.5,
            label='SFT', color=get_training_method_color('SFT'), markersize=4,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(difficulties, grpo_l_delta, marker='^', linewidth=1.5,
            label='GRPO', color=get_training_method_color('GRPO'), markersize=5,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(difficulties, sbs_delta, marker='D', linewidth=1.5,
            label='Step-by-Step', color=get_training_method_color('Step-by-step'), markersize=3.5,
            markeredgecolor='white', markeredgewidth=0.8)
    
    # Add horizontal line at y=0 (baseline)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    
    # Customize plot
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('$\\Delta$ Accuracy (\\%)')
    ax.set_xticks(difficulties)
    
    # Set yticks to include positive values
    # Calculate appropriate yticks based on data range
    all_deltas_for_ticks = sft_delta + grpo_l_delta + sbs_delta
    min_delta = min(all_deltas_for_ticks)
    max_delta = max(all_deltas_for_ticks)
    
    # Create yticks that span from negative to positive
    ytick_step = 5  # 5 percentage point steps
    ytick_min = int(min_delta / ytick_step) * ytick_step
    ytick_max = int(max_delta / ytick_step + 1) * ytick_step
    yticks = list(range(ytick_min, ytick_max + 1, ytick_step))
    ax.set_yticks(yticks)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.38, -0.25), 
              ncol=3, frameon=False, fontsize=9)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Only horizontal grid lines
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding for both positive and negative values
    all_deltas = sft_delta + grpo_l_delta + sbs_delta
    min_delta = min(all_deltas)
    max_delta = max(all_deltas)
    y_range = max_delta - min_delta
    ax.set_ylim(min_delta - 0.1 * y_range, max_delta + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save figure
    for ext in ['pdf', 'png', 'svg']:
        output_file = output_dir / f'solve_rate_by_difficulty.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - DELTA FROM BASELINE")
    print("="*80)
    print(f"\n{'Difficulty':<12} {'Baseline':<12} {'SFT Δ':<12} {'GRPO-L Δ':<12} {'Step-by-Step Δ':<12}")
    print("-"*80)
    for i, d in enumerate(difficulties):
        print(f"{d:<12} {baseline_rates.get(d, 0):>10.1f}%  {sft_delta[i]:>+10.1f}%  "
              f"{grpo_l_delta[i]:>+10.1f}%  {sbs_delta[i]:>+10.1f}%")
    
    print("\nAverage delta across all difficulties:")
    baseline_avg = np.mean([baseline_rates.get(d, 0) for d in difficulties])
    sft_delta_avg = np.mean(sft_delta)
    grpo_l_delta_avg = np.mean(grpo_l_delta)
    sbs_delta_avg = np.mean(sbs_delta)
    
    print(f"  Baseline avg: {baseline_avg:.2f}%")
    print(f"  SFT Δ:        {sft_delta_avg:+.2f}% ({sft_delta_avg / baseline_avg * 100:+.1f}% relative)")
    print(f"  GRPO-L Δ:     {grpo_l_delta_avg:+.2f}% ({grpo_l_delta_avg / baseline_avg * 100:+.1f}% relative)")
    print(f"  Step-by-Step Δ: {sbs_delta_avg:+.2f}% ({sbs_delta_avg / baseline_avg * 100:+.1f}% relative)")
    print("="*80)


if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'results' / 'figures'
    create_visualization(output_dir)
    print("\n✓ Visualization complete!")


