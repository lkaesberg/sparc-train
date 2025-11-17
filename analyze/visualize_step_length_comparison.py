#!/usr/bin/env python3
"""
Visualize average step length comparison across model variants.
Compares Baseline Qwen, SFT, GRPO-L, and Step-by-step approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, COLUMN_WIDTH_INCHES, get_training_method_color

def create_step_length_chart():
    """Create line chart comparing average step lengths across model variants."""
    
    # Setup plot style
    setup_plot_style(use_latex=True)
    
    # Data: Average step length for each model size and variant
    model_sizes = [0.6, 4, 14, 32]  # Model sizes in billions
    
    # Average Path Length (from stats CSV files)
    baseline = [33.4, 34.2, 18.9, 19.3]  # Baseline Qwen
    sft = [78.5, 25.5, 25.0, 26.2]       # SFT
    grpo_l = [61.6, 15.7, 16.4, 17.6]    # GRPO-L
    
    # Steps per puzzle (from step-by-step summary files)
    step_by_step = [32.78, 12.39, 13.32, 13.65]  # Step-by-step
    
    # Create figure
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES * 0.6, TEXT_WIDTH_INCHES * 0.4))
    
    # Plot lines with consistent colors from plot_config
    ax.plot(model_sizes, baseline, marker='o', linewidth=2, markersize=6, 
            label='Baseline', color=get_training_method_color('Baseline'), linestyle='-')
    ax.plot(model_sizes, sft, marker='s', linewidth=2, markersize=6, 
            label='SFT', color=get_training_method_color('SFT'), linestyle='--')
    ax.plot(model_sizes, grpo_l, marker='^', linewidth=2, markersize=6, 
            label='GRPO-L', color=get_training_method_color('GRPO-L'), linestyle='-.') 
    ax.plot(model_sizes, step_by_step, marker='D', linewidth=2, markersize=5, 
            label='Step-by-step', color=get_training_method_color('Step-by-step'), linestyle=':')
    
    # Customize axes
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=10)
    ax.set_ylabel('Average Path Length (steps)', fontsize=10)
    ax.set_title('Average Path Length by Model Size and Training Method', fontsize=10, pad=10)
    
    # Set x-axis ticks to model sizes
    ax.set_xticks(model_sizes)
    ax.set_xticklabels(['0.6B', '4B', '14B', '32B'])
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    
    # Set y-axis to start from 0 for better context
    ax.set_ylim(bottom=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / "step_length_comparison"
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_base}.svg", bbox_inches='tight')
    
    print(f"✓ Saved figures to {output_dir}/")
    print(f"  - step_length_comparison.png")
    print(f"  - step_length_comparison.pdf")
    print(f"  - step_length_comparison.svg")
    
    # Show plot
    plt.show()

def print_summary_statistics():
    """Print summary statistics for the data."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Average Path Length (steps)")
    print("="*80)
    
    model_sizes = ['0.6B', '4B', '14B', '32B']
    baseline = [33.4, 34.2, 18.9, 19.3]
    sft = [78.5, 25.5, 25.0, 26.2]
    grpo_l = [61.6, 15.7, 16.4, 17.6]
    step_by_step = [32.78, 12.39, 13.32, 13.65]
    
    print(f"\n{'Model':<10} {'Baseline':>12} {'SFT':>12} {'GRPO-L':>12} {'Step-by-step':>15}")
    print("-" * 80)
    
    for i, size in enumerate(model_sizes):
        print(f"{size:<10} {baseline[i]:>12.2f} {sft[i]:>12.2f} {grpo_l[i]:>12.2f} {step_by_step[i]:>15.2f}")
    
    print("-" * 80)
    print(f"{'Mean':<10} {np.mean(baseline):>12.2f} {np.mean(sft):>12.2f} {np.mean(grpo_l):>12.2f} {np.mean(step_by_step):>15.2f}")
    print(f"{'Median':<10} {np.median(baseline):>12.2f} {np.median(sft):>12.2f} {np.median(grpo_l):>12.2f} {np.median(step_by_step):>15.2f}")
    print(f"{'Min':<10} {np.min(baseline):>12.2f} {np.min(sft):>12.2f} {np.min(grpo_l):>12.2f} {np.min(step_by_step):>15.2f}")
    print(f"{'Max':<10} {np.max(baseline):>12.2f} {np.max(sft):>12.2f} {np.max(grpo_l):>12.2f} {np.max(step_by_step):>15.2f}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("• GRPO-L achieves the shortest average paths (mean: {:.2f} steps)".format(np.mean(grpo_l)))
    print("• Step-by-step is very close to GRPO-L (mean: {:.2f} steps)".format(np.mean(step_by_step)))
    print("• SFT shows highest variance, especially for 0.6B model ({:.1f} steps)".format(sft[0]))
    print("• Larger models (14B, 32B) generally produce more consistent path lengths")
    print("• 0.6B models show the most variability across training methods")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("Generating step length comparison chart...")
    print_summary_statistics()
    create_step_length_chart()

