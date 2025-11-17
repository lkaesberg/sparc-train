#!/usr/bin/env python3
"""
Visualize token counts produced by different model configurations
based on puzzle difficulty level.

This script analyzes:
- Baseline: Averaged over base Qwen models (Qwen_Qwen3-*)
- SFT: Supervised fine-tuned models
- GRPO: GRPO with long context
- Step-by-step: Results from step-by-step approach

Uses tiktoken to calculate token counts from model outputs.
"""

import json
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import seaborn as sns

# Import plot configuration
from plot_config import (
    setup_plot_style,
    COLUMN_WIDTH_INCHES,
    MODEL_COLORS,
    get_model_color
)


def count_tokens(text, model="gpt-4"):
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens in
        model: Model name for tiktoken encoding (default: gpt-4)
    
    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base which is used by GPT-4
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(str(text)))


def load_sparc_results(jsonl_path):
    """
    Load SPARC results from JSONL file.
    
    Returns:
        List of dicts with difficulty_level and token_count
    """
    results = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                difficulty = data.get('difficulty_level')
                message = data.get('result', {}).get('message', '')
                
                if difficulty and message:
                    token_count = count_tokens(message)
                    results.append({
                        'difficulty_level': difficulty,
                        'token_count': token_count,
                        'puzzle_id': data.get('id')
                    })
    
    return results


def parse_step_by_step_results(results_dir):
    """
    Parse step-by-step results from summary text files.
    
    Returns:
        Dict mapping model size to list of {difficulty_level, token_count}
    """
    step_by_step_data = {}
    
    results_path = Path(results_dir)
    
    # Pattern for parsing the summary files
    # Example: "total_completion_tokens_per_puzzle: avg=26316.08, med=25889.00, min=1832, max=67075"
    token_pattern = r'total_completion_tokens_per_puzzle: avg=([\d.]+)'
    difficulty_pattern = r'Difficulty (\d+)'
    
    for summary_file in results_path.glob('summary_by_difficulty_Qwen3-*.txt'):
        # Extract model size from filename (e.g., "0.6B", "4B", "14B", "32B")
        match = re.search(r'Qwen3-([\d.]+B)', summary_file.name)
        if not match:
            continue
        
        model_size = match.group(1)
        model_data = []
        
        with open(summary_file, 'r') as f:
            content = f.read()
            
            # Split by difficulty sections
            sections = re.split(r'Difficulty \d+', content)
            difficulties = re.findall(difficulty_pattern, content)
            
            for i, (diff, section) in enumerate(zip(difficulties, sections[1:])):
                # Extract total tokens
                token_match = re.search(token_pattern, section)
                if token_match:
                    total_tokens = float(token_match.group(1))
                    model_data.append({
                        'difficulty_level': int(diff),
                        'token_count': total_tokens
                    })
        
        if model_data:
            step_by_step_data[model_size] = model_data
    
    return step_by_step_data


def aggregate_by_configuration(sparc_dir):
    """
    Aggregate SPARC results by configuration type.
    
    Returns:
        Dict mapping configuration name to list of {difficulty_level, token_count}
    """
    sparc_path = Path(sparc_dir)
    
    configurations = {
        'Baseline': [],  # Qwen_Qwen3-* models
        'SFT': [],       # *-SFT.jsonl models
        'GRPO': []     # *-GRPO.jsonl models
    }
    
    # Process Baseline models
    for jsonl_file in sparc_path.glob('Qwen_Qwen3-*.jsonl'):
        # Skip the 1.7B and 8B models as they're not in the training set
        if '1.7B' in jsonl_file.name or '8B' in jsonl_file.name:
            continue
        
        print(f"Loading baseline: {jsonl_file.name}")
        results = load_sparc_results(jsonl_file)
        configurations['Baseline'].extend(results)
    
    # Process SFT models
    for jsonl_file in sparc_path.glob('lkaesberg_Qwen3-*-SPaRC-SFT.jsonl'):
        print(f"Loading SFT: {jsonl_file.name}")
        results = load_sparc_results(jsonl_file)
        configurations['SFT'].extend(results)
    
    # Process GRPO models
    for jsonl_file in sparc_path.glob('lkaesberg_Qwen3-*-SPaRC-GRPO-L.jsonl'):
        print(f"Loading GRPO: {jsonl_file.name}")
        results = load_sparc_results(jsonl_file)
        configurations['GRPO'].extend(results)
    
    return configurations


def compute_stats_by_difficulty(data_list):
    """
    Compute statistics for token counts grouped by difficulty level.
    
    Returns:
        Dict mapping difficulty_level to {mean, median, std, count}
    """
    df = pd.DataFrame(data_list)
    
    stats = {}
    for difficulty in sorted(df['difficulty_level'].unique()):
        diff_data = df[df['difficulty_level'] == difficulty]['token_count']
        
        stats[difficulty] = {
            'mean': diff_data.mean(),
            'median': diff_data.median(),
            'std': diff_data.std(),
            'count': len(diff_data),
            'q25': diff_data.quantile(0.25),
            'q75': diff_data.quantile(0.75)
        }
    
    return stats


def visualize_tokens_by_difficulty(
    all_stats,
    output_dir,
    show_median=False
):
    """
    Create line plot visualization of tokens produced by difficulty level.
    
    Args:
        all_stats: Dict mapping configuration name to stats dict
        output_dir: Directory to save figures
        show_median: Whether to show median instead of mean
    """
    setup_plot_style(use_latex=True)
    
    # Create figure with column width
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8))
    
    # Define colors using Seaborn Set2 palette (matching solve_rate_by_difficulty.py)
    import matplotlib.colors as mcolors
    set2_colors = plt.cm.Set2.colors
    
    # Configuration styling with enhanced visual appeal
    config_styles = {
        'Baseline': {
            'color': mcolors.rgb2hex(set2_colors[7]),  # Gray
            'marker': 'o',
            'markersize': 6,
            'linewidth': 2.0,
            'alpha': 0.25
        },
        'SFT': {
            'color': mcolors.rgb2hex(set2_colors[0]),  # Teal/green
            'marker': 's',
            'markersize': 6,
            'linewidth': 2.0,
            'alpha': 0.25
        },
        'GRPO': {
            'color': mcolors.rgb2hex(set2_colors[1]),  # Orange
            'marker': '^',
            'markersize': 7,
            'linewidth': 2.0,
            'alpha': 0.25
        },
        'Step-by-step': {
            'color': mcolors.rgb2hex(set2_colors[2]),  # Green
            'marker': 'D',
            'markersize': 5.5,
            'linewidth': 2.0,
            'alpha': 0.25
        }
    }
    
    difficulty_levels = [1, 2, 3, 4, 5]
    
    # Prepare data for line plots
    configurations = ['Baseline', 'SFT', 'GRPO', 'Step-by-step']
    configurations = [c for c in configurations if c in all_stats and all_stats[c]]
    
    # Plot lines for each configuration with shaded error regions
    for config_name in configurations:
        stats = all_stats[config_name]
        
        values = []
        stds = []
        q25s = []
        q75s = []
        
        for diff in difficulty_levels:
            if diff in stats:
                if show_median:
                    values.append(stats[diff]['median'])
                    # For median, use IQR for shading
                    q25s.append(stats[diff]['q25'])
                    q75s.append(stats[diff]['q75'])
                else:
                    values.append(stats[diff]['mean'])
                    stds.append(stats[diff]['std'])
            else:
                values.append(np.nan)
                if show_median:
                    q25s.append(np.nan)
                    q75s.append(np.nan)
                else:
                    stds.append(np.nan)
        
        style = config_styles.get(config_name, {})
        color = style.get('color', '#999999')
        
        # Plot shaded error region first (so it's behind the line)
        if show_median:
            # Shade between Q1 and Q3 (IQR)
            ax.fill_between(
                difficulty_levels,
                q25s,
                q75s,
                alpha=style.get('alpha', 0.25),
                color=color,
                linewidth=0
            )
        else:
            # Shade ±1 standard deviation
            lower = np.array(values) - np.array(stds)
            upper = np.array(values) + np.array(stds)
            ax.fill_between(
                difficulty_levels,
                lower,
                upper,
                alpha=style.get('alpha', 0.25),
                color=color,
                linewidth=0
            )
        
        # Plot line with markers on top
        ax.plot(
            difficulty_levels,
            values,
            label=config_name,
            color=color,
            marker=style.get('marker', 'o'),
            markersize=style.get('markersize', 6),
            linewidth=style.get('linewidth', 2.0),
            markeredgecolor='white',
            markeredgewidth=1.2,
            zorder=3
        )
    
    # Formatting
    ax.set_xlabel('Difficulty Level', fontsize=10, fontweight='normal')
    ax.set_ylabel('Tokens per Answer', fontsize=10, fontweight='normal')
    ax.set_xticks(difficulty_levels)
    
    # Legend at bottom center with better styling
    legend = ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.22), 
        ncol=2, 
        frameon=False, 
        fontsize=9,
        columnspacing=1.5,
        handletextpad=0.5
    )
    
    # Make legend markers larger
    for handle in legend.legend_handles:
        handle.set_markersize(8)
        handle.set_linewidth(2.5)
    
    # Remove top and right spines, make bottom and left thicker
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Enhanced grid styling
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add some padding to y-axis
    all_values = []
    for config_name in configurations:
        stats = all_stats[config_name]
        for diff in difficulty_levels:
            if diff in stats:
                if show_median:
                    all_values.append(stats[diff]['median'])
                else:
                    all_values.append(stats[diff]['mean'])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.1 * y_range)
    
    # Format y-axis to use thousands separator
    from matplotlib.ticker import FuncFormatter
    def thousands_formatter(x, pos):
        return f'{int(x):,}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metric = 'median' if show_median else 'mean'
    
    for ext in ['pdf', 'png', 'svg']:
        filename = f'tokens_by_difficulty_{metric}.{ext}'
        fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / filename}")
    
    plt.close()


def print_summary_table(all_stats):
    """
    Print a summary table of token statistics.
    """
    print("\n" + "="*80)
    print("TOKEN STATISTICS BY CONFIGURATION AND DIFFICULTY")
    print("="*80)
    
    for config_name, stats in all_stats.items():
        if not stats:
            continue
        
        print(f"\n{config_name}:")
        print("-" * 80)
        print(f"{'Difficulty':<12} {'Mean':<12} {'Median':<12} {'Std':<12} {'Count':<12}")
        print("-" * 80)
        
        for diff in sorted(stats.keys()):
            s = stats[diff]
            print(f"{diff:<12} {s['mean']:<12.1f} {s['median']:<12.1f} "
                  f"{s['std']:<12.1f} {s['count']:<12}")
    
    print("="*80)


def save_csv_summary(all_stats, output_dir):
    """
    Save summary statistics to CSV file.
    """
    rows = []
    
    for config_name, stats in all_stats.items():
        if not stats:
            continue
        
        for diff in sorted(stats.keys()):
            s = stats[diff]
            rows.append({
                'configuration': config_name,
                'difficulty': diff,
                'mean_tokens': s['mean'],
                'median_tokens': s['median'],
                'std_tokens': s['std'],
                'q25_tokens': s['q25'],
                'q75_tokens': s['q75'],
                'sample_count': s['count']
            })
    
    df = pd.DataFrame(rows)
    output_path = Path(output_dir) / 'tokens_by_difficulty_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary CSV: {output_path}")


def create_heatmap(all_stats, output_dir, show_median=False, annotate=True):
    """
    Create heatmap showing token counts by configuration and difficulty.
    
    Args:
        all_stats: Dict mapping configuration name to stats dict
        output_dir: Directory to save figures
        show_median: Whether to show median instead of mean
        annotate: Whether to print numeric values in each cell
    """
    setup_plot_style(use_latex=True)
    
    # Create figure with better proportions - wider and taller for better cell sizing
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES * 1.2, COLUMN_WIDTH_INCHES * 0.6))
    
    # Prepare data for heatmap
    difficulty_levels = [1, 2, 3, 4, 5]
    configurations = ['Baseline', 'SFT', 'GRPO', 'Step-by-step']
    configurations = [c for c in configurations if c in all_stats and all_stats[c]]
    
    # Create display labels (abbreviate Step-by-step)
    config_labels = []
    for config in configurations:
        if config == 'Step-by-step':
            config_labels.append('Step-by-step')
        else:
            config_labels.append(config)
    
    # Build matrix
    matrix = []
    for config_name in configurations:
        stats = all_stats[config_name]
        row = []
        for diff in difficulty_levels:
            if diff in stats:
                if show_median:
                    row.append(stats[diff]['median'])
                else:
                    row.append(stats[diff]['mean'])
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap with seaborn
    # Use a perceptually uniform colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Create heatmap with proper aspect ratio
    im = ax.imshow(matrix, cmap=cmap, aspect='equal')
    
    # Set ticks with better spacing
    ax.set_xticks(np.arange(len(difficulty_levels)))
    ax.set_yticks(np.arange(len(configurations)))
    ax.set_xticklabels(difficulty_levels, fontsize=9)
    ax.set_yticklabels(config_labels, fontsize=9)
    
    # Labels (only x-axis, no y-axis label)
    ax.set_xlabel('Difficulty Level', fontsize=9)
    
    # Add annotations if requested
    if annotate:
        for i in range(len(configurations)):
            for j in range(len(difficulty_levels)):
                value = matrix[i, j]
                if not np.isnan(value):
                    # Format value in thousands with one decimal place (e.g., 10432 -> 10.4)
                    text = f'{value/1000:.1f}'
                    
                    # Choose text color based on background
                    # Use white text for dark backgrounds, black for light
                    text_color = 'white' if value > (matrix.max() * 0.6) else 'black'
                    
                    ax.text(j, i, text,
                           ha="center", va="center",
                           color=text_color, fontsize=8, fontweight='normal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(r'Tokens per Answer ($\times$1000)', rotation=270, labelpad=12, fontsize=8)
    
    # Format colorbar ticks in thousands with one decimal
    # Use FuncFormatter to properly format the colorbar
    from matplotlib.ticker import FuncFormatter
    def format_thousands(x, pos):
        return f'{x/1000:.1f}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    cbar.ax.tick_params(labelsize=7)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add minor ticks for grid
    ax.set_xticks(np.arange(len(difficulty_levels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(configurations)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metric = 'median' if show_median else 'mean'
    annot_suffix = '_annotated' if annotate else ''
    
    for ext in ['pdf', 'png', 'svg']:
        filename = f'tokens_by_difficulty_heatmap_{metric}{annot_suffix}.{ext}'
        fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / filename}")
    
    plt.close()


def create_efficiency_plot(all_stats, sparc_dir, step_by_step_dir, output_dir):
    """
    Create efficiency plot: Tokens vs Accuracy.
    Shows trade-off between token usage and solve rate.
    
    Args:
        all_stats: Dict mapping configuration name to stats dict
        sparc_dir: Path to SPARC results directory
        step_by_step_dir: Path to step-by-step results directory
        output_dir: Directory to save figures
    """
    setup_plot_style(use_latex=True)
    
    # Import for parsing stats
    import re
    
    # Helper function to parse solve rates from stats CSV
    def parse_solve_rates_csv(csv_path):
        solve_rates = {}
        with open(csv_path, 'r') as f:
            for line in f:
                match = re.match(r'Difficulty (\d+) Solved,\d+/\d+,([\d.]+)%', line)
                if match:
                    difficulty = int(match.group(1))
                    percentage = float(match.group(2))
                    solve_rates[difficulty] = percentage
        return solve_rates
    
    # Helper function to parse solve rates from step-by-step summary
    def parse_solve_rates_sbs(txt_path):
        solve_rates = {}
        with open(txt_path, 'r') as f:
            content = f.read()
        pattern = r'Difficulty (\d+).*?wins:\s+([\d.]+)%'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            difficulty = int(match[0])
            percentage = float(match[1])
            solve_rates[difficulty] = percentage
        return solve_rates
    
    # Collect data points
    data_points = []
    
    # Process baseline models (average across model sizes)
    base_models = ['Qwen_Qwen3-0.6B', 'Qwen_Qwen3-4B', 'Qwen_Qwen3-14B', 'Qwen_Qwen3-32B']
    for difficulty in [1, 2, 3, 4, 5]:
        rates = []
        for model in base_models:
            csv_path = Path(sparc_dir) / f'{model}_stats.csv'
            if csv_path.exists():
                solve_rates = parse_solve_rates_csv(csv_path)
                if difficulty in solve_rates:
                    rates.append(solve_rates[difficulty])
        
        if rates and difficulty in all_stats.get('Baseline', {}):
            avg_rate = np.mean(rates)
            tokens = all_stats['Baseline'][difficulty]['mean']
            data_points.append({
                'config': 'Baseline',
                'difficulty': difficulty,
                'accuracy': avg_rate,
                'tokens': tokens
            })
    
    # Process SFT models (average across model sizes)
    sft_models = ['lkaesberg_Qwen3-0.6B-SPaRC-SFT', 'lkaesberg_Qwen3-4B-SPaRC-SFT', 
                  'lkaesberg_Qwen3-14B-SPaRC-SFT', 'lkaesberg_Qwen3-32B-SPaRC-SFT']
    for difficulty in [1, 2, 3, 4, 5]:
        rates = []
        for model in sft_models:
            csv_path = Path(sparc_dir) / f'{model}_stats.csv'
            if csv_path.exists():
                solve_rates = parse_solve_rates_csv(csv_path)
                if difficulty in solve_rates:
                    rates.append(solve_rates[difficulty])
        
        if rates and difficulty in all_stats.get('SFT', {}):
            avg_rate = np.mean(rates)
            tokens = all_stats['SFT'][difficulty]['mean']
            data_points.append({
                'config': 'SFT',
                'difficulty': difficulty,
                'accuracy': avg_rate,
                'tokens': tokens
            })
    
    # Process GRPO models (average across model sizes)
    grpo_models = ['lkaesberg_Qwen3-0.6B-SPaRC-GRPO-L', 'lkaesberg_Qwen3-4B-SPaRC-GRPO-L',
                   'lkaesberg_Qwen3-14B-SPaRC-GRPO-L', 'lkaesberg_Qwen3-32B-SPaRC-GRPO-L']
    for difficulty in [1, 2, 3, 4, 5]:
        rates = []
        for model in grpo_models:
            csv_path = Path(sparc_dir) / f'{model}_stats.csv'
            if csv_path.exists():
                solve_rates = parse_solve_rates_csv(csv_path)
                if difficulty in solve_rates:
                    rates.append(solve_rates[difficulty])
        
        if rates and difficulty in all_stats.get('GRPO', {}):
            avg_rate = np.mean(rates)
            tokens = all_stats['GRPO'][difficulty]['mean']
            data_points.append({
                'config': 'GRPO',
                'difficulty': difficulty,
                'accuracy': avg_rate,
                'tokens': tokens
            })
    
    # Process step-by-step models (average across model sizes)
    sbs_models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B', 'Qwen3-32B']
    for difficulty in [1, 2, 3, 4, 5]:
        rates = []
        for model in sbs_models:
            txt_path = Path(step_by_step_dir) / f'summary_by_difficulty_{model}.txt'
            if txt_path.exists():
                solve_rates = parse_solve_rates_sbs(txt_path)
                if difficulty in solve_rates:
                    rates.append(solve_rates[difficulty])
        
        if rates and difficulty in all_stats.get('Step-by-step', {}):
            avg_rate = np.mean(rates)
            tokens = all_stats['Step-by-step'][difficulty]['mean']
            data_points.append({
                'config': 'Step-by-step',
                'difficulty': difficulty,
                'accuracy': avg_rate,
                'tokens': tokens
            })
    
    # Create DataFrame and aggregate across all difficulties
    df = pd.DataFrame(data_points)
    
    # Aggregate by configuration (average across all difficulties)
    df_agg = df.groupby('config').agg({
        'accuracy': 'mean',
        'tokens': 'mean'
    }).reset_index()
    
    # Create figure with reduced height
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.8))
    
    # Define colors and markers using plot_config.py colors
    from plot_config import get_training_method_color
    
    config_styles = {
        'Baseline': {
            'color': get_training_method_color('Baseline'),
            'marker': 'o',
        },
        'SFT': {
            'color': get_training_method_color('SFT'),
            'marker': 's',
        },
        'GRPO': {
            'color': get_training_method_color('GRPO'),
            'marker': '^',
        },
        'Step-by-step': {
            'color': get_training_method_color('Step-by-step'),
            'marker': 'D',
        }
    }
    
    # Plot each configuration (one point per configuration)
    for config in ['Baseline', 'SFT', 'GRPO', 'Step-by-step']:
        config_data = df_agg[df_agg['config'] == config]
        if len(config_data) == 0:
            continue
        
        style = config_styles.get(config, {})
        
        # Get the single point for this configuration
        accuracy = config_data['accuracy'].values[0]
        tokens = config_data['tokens'].values[0]
        
        # Plot marker (no transparency) - swapped x and y
        ax.scatter(
            tokens,
            accuracy,
            s=80,
            color=style.get('color'),
            marker=style.get('marker', 'o'),
            alpha=1.0,
            edgecolors='white',
            linewidths=1.5,
            label=config,
            zorder=3
        )
        
        # Add label next to the marker (closer) - adjusted for swapped axes
        ax.text(
            tokens,
            accuracy + 0.5,  # offset upward
            config,
            fontsize=9,
            ha='center',
            va='bottom',
            color='black',
            fontweight='normal',
            zorder=4
        )
    
    # Formatting - swapped x and y labels
    ax.set_xlabel(r'Tokens per Answer ($\times$1000)', fontsize=10)
    ax.set_ylabel('Accuracy (\\%)', fontsize=10)
    #ax.set_xscale('log')
    
    # Set x and y limits to start at 0
    ax.set_xlim(0, 26000)
    ax.set_ylim(0, None)
    
    # Format x-axis to show ALL values in thousands with one decimal
    from matplotlib.ticker import FuncFormatter
    def format_thousands_x(x, pos):
        return f'{x/1000:.1f}'
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands_x))
    
    # No legend needed since we have labels next to points
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
    #           ncol=2, frameon=False, fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_axisbelow(True)
    
    # Extend y-axis limits for better spacing
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.1 * y_range)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for ext in ['pdf', 'png', 'svg']:
        filename = f'tokens_vs_accuracy_efficiency.{ext}'
        fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / filename}")
    
    plt.close()


def create_scatter_plot(configurations, output_dir):
    """
    Create scatter plot showing token distribution by configuration,
    colored by difficulty level.
    
    Args:
        configurations: Dict mapping configuration name to list of {difficulty_level, token_count}
        output_dir: Directory to save figures
    """
    setup_plot_style(use_latex=True)
    
    # Prepare data for scatter plot
    data_rows = []
    for config_name, data_list in configurations.items():
        for entry in data_list:
            data_rows.append({
                'Configuration': config_name,
                'Tokens': entry['token_count'],
                'Difficulty': entry['difficulty_level']
            })
    
    df = pd.DataFrame(data_rows)
    
    # Create figure with column width
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.9))
    
    # Define color palette for difficulty levels (5 levels)
    # Use a sequential colormap that goes from light to dark
    difficulty_colors = sns.color_palette("YlOrRd", n_colors=5)
    
    # Order configurations
    config_order = ['Baseline', 'SFT', 'GRPO', 'Step-by-step']
    config_order = [c for c in config_order if c in df['Configuration'].unique()]
    
    # Create scatter plot with seaborn
    sns.stripplot(
        data=df,
        x='Configuration',
        y='Tokens',
        hue='Difficulty',
        palette=difficulty_colors,
        order=config_order,
        hue_order=[1, 2, 3, 4, 5],
        alpha=0.6,
        size=2,
        jitter=0.25,
        ax=ax,
        dodge=False,
        legend='auto'
    )
    
    # Formatting
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Tokens per Answer')
    
    # Rotate x-axis labels if needed
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Move legend to bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Difficulty', 
              loc='upper center', bbox_to_anchor=(0.5, -0.25),
              ncol=5, frameon=False, fontsize=8, title_fontsize=9)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal grid lines
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for ext in ['pdf', 'png', 'svg']:
        filename = f'tokens_by_difficulty_scatter.{ext}'
        fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / filename}")
    
    plt.close()


def main():
    """Main execution function."""
    
    # Paths
    base_dir = Path(__file__).parent
    sparc_dir = base_dir / 'results' / 'sparc'
    step_by_step_dir = base_dir / 'results' / 'step-by-step'
    output_dir = base_dir / 'results' / 'figures'
    
    print("="*80)
    print("TOKEN ANALYSIS BY PUZZLE DIFFICULTY")
    print("="*80)
    
    # Load and aggregate SPARC results
    print("\n1. Loading SPARC results...")
    configurations = aggregate_by_configuration(sparc_dir)
    
    # Load step-by-step results
    print("\n2. Loading step-by-step results...")
    step_by_step_data = parse_step_by_step_results(step_by_step_dir)
    
    # Average step-by-step across model sizes
    if step_by_step_data:
        print(f"   Found step-by-step data for models: {list(step_by_step_data.keys())}")
        
        # Combine all step-by-step data
        all_step_data = []
        for model_size, data in step_by_step_data.items():
            for entry in data:
                all_step_data.append(entry)
        
        configurations['Step-by-step'] = all_step_data
    
    # Compute statistics for each configuration
    print("\n3. Computing statistics...")
    all_stats = {}
    
    for config_name, data_list in configurations.items():
        if data_list:
            print(f"   {config_name}: {len(data_list)} samples")
            all_stats[config_name] = compute_stats_by_difficulty(data_list)
        else:
            print(f"   {config_name}: No data found")
    
    # Print summary table
    print_summary_table(all_stats)
    
    # Save CSV summary
    save_csv_summary(all_stats, output_dir)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # Line plot with mean
    visualize_tokens_by_difficulty(
        all_stats,
        output_dir,
        show_median=False
    )
    
    # Line plot with median
    visualize_tokens_by_difficulty(
        all_stats,
        output_dir,
        show_median=True
    )
    
    # Heatmap visualizations
    print("\n5. Creating heatmaps...")
    
    # Heatmap with mean values (annotated)
    create_heatmap(all_stats, output_dir, show_median=False, annotate=True)
    
    # Heatmap with median values (annotated)
    create_heatmap(all_stats, output_dir, show_median=True, annotate=True)
    
    # Heatmap without annotations (cleaner look)
    create_heatmap(all_stats, output_dir, show_median=False, annotate=False)
    
    # Efficiency plot: Tokens vs Accuracy
    print("\n6. Creating efficiency plot...")
    create_efficiency_plot(all_stats, sparc_dir, step_by_step_dir, output_dir)
    
    # Scatter plot showing all individual samples
    print("\n7. Creating scatter plot...")
    create_scatter_plot(configurations, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()




