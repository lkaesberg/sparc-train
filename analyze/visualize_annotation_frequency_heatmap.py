#!/usr/bin/env python3
"""
Visualize annotation frequency heatmap comparing majority vote (≥2 human annotators)
against model predictions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, COLUMN_WIDTH_INCHES

# Mapping from human codes to letter codes
HUMAN_TO_LETTER = {
    'a_planning_logical_flaw': 'A',
    'b_misunderstood_invented_rule': 'B',
    'c_spatial_geometric_misjudgment': 'C',
    'd_premature_verification': 'D',
    'e_no_correction_despite_noticing': 'E',
    'f_grid_coordinate_error': 'F',
}

CATEGORY_LABELS = {
    'A': 'Planning',
    'B': 'Misunderstood Rule',
    'C': 'Spatial/Geometric',
    'D': 'Premature Verification',
    'E': 'No Correction',
    'F': 'Coordinate Error',
}

def load_jsonl(filepath):
    """Load JSONL file and return list of JSON objects."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_human_annotations(sample):
    """Extract failure categories from human annotation."""
    if 'failure_annotation' not in sample:
        return set()
    
    failure_reasons = sample['failure_annotation'].get('failure_reasons', [])
    # Convert to letter codes
    letter_codes = {HUMAN_TO_LETTER[code] for code in failure_reasons if code in HUMAN_TO_LETTER}
    return letter_codes


def extract_llm_annotations(sample):
    """Extract failure categories from LLM annotation."""
    if 'llm_annotation' not in sample:
        return set()
    
    categories = sample['llm_annotation'].get('categories', [])
    return set(categories)


def calculate_majority_vote(human_data_list):
    """
    Calculate majority vote (>=2 annotators agree) for each puzzle.
    
    Args:
        human_data_list: List of lists of annotation data from different annotators
        
    Returns:
        dict: {puzzle_id: set of categories with majority vote}
    """
    # Organize annotations by puzzle_id
    annotations_by_puzzle = defaultdict(list)
    
    for annotator_data in human_data_list:
        for sample in annotator_data:
            puzzle_id = sample['id']
            categories = extract_human_annotations(sample)
            annotations_by_puzzle[puzzle_id].append(categories)
    
    # Calculate majority vote for each puzzle
    majority_votes = {}
    for puzzle_id, annotator_sets in annotations_by_puzzle.items():
        # Count how many annotators assigned each category
        category_counts = Counter()
        for category_set in annotator_sets:
            for category in category_set:
                category_counts[category] += 1
        
        # Keep categories with ≥2 votes
        majority_categories = {cat for cat, count in category_counts.items() if count >= 2}
        majority_votes[puzzle_id] = majority_categories
    
    return majority_votes


def calculate_category_frequencies(annotations_dict, exclude_categories=None):
    """
    Calculate frequency of each category.
    
    Args:
        annotations_dict: {puzzle_id: set of categories}
        exclude_categories: set of categories to exclude (e.g., {'A'})
        
    Returns:
        dict: {category: frequency (proportion)}
    """
    if exclude_categories is None:
        exclude_categories = set()
    
    category_counts = Counter()
    total_annotations = 0
    
    for categories in annotations_dict.values():
        for category in categories:
            if category not in exclude_categories:
                category_counts[category] += 1
                total_annotations += 1
    
    # Calculate frequencies (normalized)
    all_categories = ['A', 'B', 'C', 'D', 'E', 'F']
    frequencies = {}
    for cat in all_categories:
        if cat not in exclude_categories:
            frequencies[cat] = category_counts.get(cat, 0) / max(total_annotations, 1)
    
    return frequencies


def load_model_annotations(annotate_dir, model_files=None):
    """
    Load model annotations from the annotate directory.
    
    Args:
        annotate_dir: Path to the annotate directory
        model_files: Optional list of specific model files to load
        
    Returns:
        dict: {model_name: {puzzle_id: set of categories}}
    """
    annotate_path = Path(annotate_dir)
    
    # Mapping from internal names to display names
    MODEL_NAME_MAPPING = {
        'openai gpt oss 120b': 'GPT-OSS 120B',
        'openai gpt oss 20b': 'GPT-OSS 20B',
        'Qwen Qwen3 32B': 'Qwen 3 32B',
        'lkaesberg Qwen3 32B SPaRC GRPO L': 'SPaRC GRPO 32B',
        'google gemma 3 27b it': 'Gemma 3 27B',
        'meta llama Llama 3.3 70B Instruct': 'Llama 3.3 70B',
        'RedHatAI Llama 4 Maverick 17B 128E': 'Llama 4 Maverick',
        'deepseek ai DeepSeek R1 Distill Llama 70B': 'Deepseek R1 70B',
        'RedHatAI Llama 4 Scout 17B 16E': 'Llama 4 Scout',
        'Qwen Qwen2.5 72B Instruct': 'Qwen 2.5 72B',
        'deepseek ai DeepSeek R1 Distill Qwen 32B': 'Deepseek R1 32B',
    }
    
    # If no specific files provided, load all annotation_samples files
    if model_files is None:
        model_files = list(annotate_path.glob('annotation_samples.annotated_by_*.jsonl'))
    else:
        model_files = [annotate_path / f for f in model_files]
    
    model_annotations = {}
    
    for filepath in model_files:
        # Extract model name from filename
        filename = filepath.name
        if filename.startswith('annotation_samples.annotated_by_'):
            model_name = filename.replace('annotation_samples.annotated_by_', '').replace('.jsonl', '')
            # Clean up model name for display
            model_name = model_name.replace('_', ' ').replace('-', ' ')
        else:
            model_name = filepath.stem
        
        # Apply name mapping
        display_name = MODEL_NAME_MAPPING.get(model_name, model_name)
        
        # Load annotations
        data = load_jsonl(filepath)
        annotations = {}
        for sample in data:
            puzzle_id = sample['id']
            categories = extract_llm_annotations(sample)
            annotations[puzzle_id] = categories
        
        model_annotations[display_name] = annotations
    
    return model_annotations


def get_model_logo(model_name, logos_dir):
    """
    Get logo image for a model.
    
    Args:
        model_name: Name of the model
        logos_dir: Path to logos directory
        
    Returns:
        OffsetImage or None if no logo found
    """
    # Logo configuration: (width, height, zoom)
    LOGO_CONFIG = {
        "gemma.png": (64, 64, 1/7),
        "qwen.png": (64, 64, 1/7),
        "deepseek.png": (64, 64, 1/6),
        "llama.png": (64, 64, 1/6),
        "human.png": (64, 64, 1/8),
        "openai.png": (64, 64, 1/7),
    }
    
    LOGO_MAPPING = {
        "Human": "human.png",
        "Gemma": "gemma.png",
        "Qwen": "qwen.png",
        "Deepseek": "deepseek.png",
        "Llama": "llama.png",
        "GPT-OSS": "openai.png",
        "SPaRC": "qwen.png",  # SPaRC is based on Qwen
    }
    
    logo_path = None
    for keyword, logo in LOGO_MAPPING.items():
        if keyword in model_name:
            logo_path = logos_dir / logo
            if logo_path.exists():
                break
    
    if not logo_path or not logo_path.exists():
        return None
    
    # Load the logo with PIL
    img_pil = Image.open(str(logo_path)).convert('RGBA')
    
    # Get configuration (size and zoom) based on logo filename
    width, height, zoom = LOGO_CONFIG.get(logo_path.name, (64, 64, 1/9))
    
    # Resize to thumbnail size while maintaining aspect ratio
    img_pil.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    # Create and return OffsetImage with specified zoom
    imagebox = OffsetImage(np.array(img_pil), zoom=zoom)
    
    return imagebox


def create_frequency_heatmap(human_majority_freq, model_frequencies, output_path, exclude_categories=None):
    """
    Create a heatmap comparing human majority vote frequencies with model frequencies.
    
    Args:
        human_majority_freq: dict of {category: frequency} for human majority vote
        model_frequencies: dict of {model_name: {category: frequency}}
        output_path: Path to save the figure
        exclude_categories: set of categories to exclude (e.g., {'A'})
    """
    setup_plot_style(use_latex=True)
    
    if exclude_categories is None:
        exclude_categories = set()
    
    # Filter categories
    all_categories = ['A', 'B', 'C', 'D', 'E', 'F']
    categories = [cat for cat in all_categories if cat not in exclude_categories]
    
    # Define desired order for models
    model_order = [
        'GPT-OSS 120B',
        'GPT-OSS 20B',
        'Qwen 3 32B',
        'SPaRC GRPO 32B',
        'Gemma 3 27B',
        'Llama 3.3 70B',
        'Llama 4 Maverick',
        'Deepseek R1 70B',
        'Llama 4 Scout',
        'Qwen 2.5 72B',
        'Deepseek R1 32B',
    ]
    
    # Sort models according to the desired order
    sorted_model_names = [m for m in model_order if m in model_frequencies]
    # Add any models not in the order list at the end
    for m in model_frequencies:
        if m not in sorted_model_names:
            sorted_model_names.append(m)
    
    # Prepare data for heatmap
    # Rows: Human + Model names (sorted)
    row_labels = [r'Human'] + sorted_model_names
    
    # Build frequency matrix
    freq_matrix = []
    
    # Add human majority vote row
    human_row = [human_majority_freq.get(cat, 0) for cat in categories]
    freq_matrix.append(human_row)
    
    # Add model rows in sorted order
    for model_name in sorted_model_names:
        model_row = [model_frequencies[model_name].get(cat, 0) for cat in categories]
        freq_matrix.append(model_row)
    
    freq_matrix = np.array(freq_matrix)
    
    # Create figure with square cells
    n_rows = len(row_labels)
    n_cols = len(categories)
    cell_size = 0.8  # inches per cell
    fig_width = n_cols * cell_size
    fig_height = n_rows * cell_size
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap with spacing
    # We'll use imshow with interpolation='nearest' and add spacing via the grid
    im = ax.imshow(freq_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=freq_matrix.max(), 
                   interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([CATEGORY_LABELS[cat] for cat in categories], fontsize=9, 
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(row_labels, fontsize=9)
    
    # Add y-axis label
    ax.set_ylabel('Annotators', fontsize=10, labelpad=10)
    
    # Move y-axis labels to the left to make room for logos
    ax.tick_params(axis='y', pad=15)
    
    # Remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Frequency', rotation=270, labelpad=15, fontsize=10)
    
    # Add spacing between cells using grid
    ax.set_xticks(np.arange(len(categories)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Remove border (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add logos to y-axis
    logos_dir = Path(__file__).parent / 'images' / 'logos'
    yticks = np.arange(len(row_labels))
    if logos_dir.exists():
        for i, model_name in enumerate(row_labels):
            logo = get_model_logo(model_name, logos_dir)
            if logo:
                # Position the icon to the left of the y-tick labels
                # Using axis coordinates for proper positioning
                ab = AnnotationBbox(
                    logo,
                    xy=(0, yticks[i]),  # Use the actual tick position
                    xycoords=('axes fraction', 'data'),  # x in axes coords, y in data coords
                    xybox=(-8, 0),  # Offset to the left of the axis
                    boxcoords="offset points",
                    frameon=False,
                    box_alignment=(0.5, 0.5),  # Center the icon
                    zorder=10
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    
    print(f"✓ Saved heatmap to {output_path.with_suffix('.pdf')}")
    print(f"✓ Saved heatmap to {output_path.with_suffix('.png')}")
    print(f"✓ Saved heatmap to {output_path.with_suffix('.svg')}")
    
    plt.close()


def main():
    # Paths
    results_dir = Path(__file__).parent / 'results'
    human_annotation_dir = results_dir / 'human_annotation'
    annotate_dir = results_dir / 'annotate'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Load human annotations
    print("Loading human annotations...")
    human_annotators = [
        'abukhanov_sparc_annotated.jsonl',
        'hagenkort_sparc_annotated.jsonl',
        'juharova_sparc_annotated.jsonl',
    ]
    
    human_data_list = []
    for annotator_file in human_annotators:
        filepath = human_annotation_dir / annotator_file
        data = load_jsonl(filepath)
        human_data_list.append(data)
        print(f"  Loaded {len(data)} samples from {annotator_file}")
    
    # Calculate majority vote
    print("\nCalculating majority vote (>=2/3 annotators)...")
    majority_votes = calculate_majority_vote(human_data_list)
    print(f"  Found {len(majority_votes)} puzzles with annotations")
    
    # Exclude category A
    exclude_categories = {'A'}
    
    # Calculate frequencies for majority vote
    human_majority_freq = calculate_category_frequencies(majority_votes, exclude_categories=exclude_categories)
    print("\nHuman Majority (>=2/3) frequencies (excluding A):")
    for cat in ['B', 'C', 'D', 'E', 'F']:
        print(f"  {cat}: {human_majority_freq[cat]:.3f}")
    
    # Load model annotations - all files starting with annotation_samples
    print("\nLoading model annotations...")
    # Get all annotation_samples files
    model_files = sorted(annotate_dir.glob('annotation_samples.annotated_by_*.jsonl'))
    
    model_annotations = load_model_annotations(annotate_dir, [f.name for f in model_files])
    print(f"  Loaded {len(model_annotations)} models")
    
    # Calculate frequencies for each model
    model_frequencies = {}
    for model_name, annotations in model_annotations.items():
        freq = calculate_category_frequencies(annotations, exclude_categories=exclude_categories)
        model_frequencies[model_name] = freq
        print(f"\n{model_name} frequencies:")
        for cat in ['B', 'C', 'D', 'E', 'F']:
            print(f"  {cat}: {freq[cat]:.3f}")
    
    # Create heatmap
    print("\nCreating frequency heatmap...")
    output_path = figures_dir / 'annotation_frequency_heatmap_majority_vs_models'
    create_frequency_heatmap(human_majority_freq, model_frequencies, output_path, 
                           exclude_categories=exclude_categories)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

