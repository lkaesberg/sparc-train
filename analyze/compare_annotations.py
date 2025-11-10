#!/usr/bin/env python3
"""
Compare human annotations with machine (LLM) annotations using macro F1 score.

This script calculates the macro F1 score between human failure annotations
and machine (LLM) annotations for SPaRC puzzle results.

The script compares annotations by:
1. Loading human annotations from analyze/results/human_annotation/hagenkortjuharova_sparc_annotated.jsonl
2. Finding all machine annotation files with prefix "annotation_samples" in analyze/results/annotate/
3. Matching samples by line number (validates that puzzle IDs match at each line)
4. Converting human annotation codes (a_*, b_*, etc.) to LLM letter codes (A, B, etc.)
5. Calculating macro F1 score across all 6 failure categories (A-F)

Failure categories:
  A: Planning/logical flaw in the reasoning approach
  B: Misunderstood or invented puzzle rules
  C: Spatial/geometric misjudgment or miscalculation
  D: Premature verification - claims correctness without checking key rules
  E: No correction despite noticing - recognizes errors but doesn't adjust
  F: Grid/coordinate error - off-by-one, swapped x/y, or out-of-bounds steps

Output:
- Macro F1, precision, and recall (averaged across all categories)
- Per-category F1, precision, recall, and support
- Summary table comparing all machine annotation files

Usage:
    python analyze/compare_annotations.py
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Mapping from human annotation codes to LLM letter codes
HUMAN_TO_LLM_MAPPING = {
    'a_planning_logical_flaw': 'A',
    'b_misunderstood_invented_rule': 'B',
    'c_spatial_geometric_misjudgment': 'C',
    'd_premature_verification': 'D',
    'e_no_correction_despite_noticing': 'E',
    'f_grid_coordinate_error': 'F',
}

# All possible categories
ALL_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F']


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_human_annotations(sample: Dict) -> Set[str]:
    """Extract human failure annotations and convert to LLM letter codes."""
    if 'failure_annotation' not in sample:
        return set()
    
    failure_reasons = sample['failure_annotation'].get('failure_reasons', [])
    
    # Convert human codes to LLM letter codes
    llm_codes = set()
    for reason in failure_reasons:
        if reason in HUMAN_TO_LLM_MAPPING:
            llm_codes.add(HUMAN_TO_LLM_MAPPING[reason])
    
    return llm_codes


def extract_llm_annotations(sample: Dict) -> Set[str]:
    """Extract LLM annotations."""
    if 'llm_annotation' not in sample:
        return set()
    
    categories = sample['llm_annotation'].get('categories', [])
    return set(categories)


def calculate_binary_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_metrics(human_annotations: List[Set[str]], 
                      llm_annotations: List[Set[str]],
                      all_categories: List[str]) -> Dict:
    """
    Calculate macro F1, precision, and recall scores.
    
    Args:
        human_annotations: List of sets containing human annotation categories
        llm_annotations: List of sets containing LLM annotation categories
        all_categories: List of all possible category codes
    
    Returns:
        Dictionary with metrics
    """
    n_samples = len(human_annotations)
    
    # Calculate per-category metrics
    per_category_metrics = {}
    
    for category in all_categories:
        # Create binary vectors for this category
        y_true = [1 if category in human_cats else 0 for human_cats in human_annotations]
        y_pred = [1 if category in llm_cats else 0 for llm_cats in llm_annotations]
        
        precision, recall, f1 = calculate_binary_metrics(y_true, y_pred)
        support = sum(y_true)
        
        per_category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # Calculate macro-averaged metrics (average across categories)
    macro_precision = sum(m['precision'] for m in per_category_metrics.values()) / len(all_categories)
    macro_recall = sum(m['recall'] for m in per_category_metrics.values()) / len(all_categories)
    macro_f1 = sum(m['f1'] for m in per_category_metrics.values()) / len(all_categories)
    
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_category': per_category_metrics,
        'n_samples': n_samples
    }


def compare_annotations(human_file: Path, machine_file: Path) -> Dict:
    """
    Compare human and machine annotations from two JSONL files.
    Matches samples by line number and validates that IDs match.
    
    Args:
        human_file: Path to human annotations file
        machine_file: Path to machine annotations file
    
    Returns:
        Dictionary with comparison metrics
    
    Raises:
        ValueError: If IDs don't match at the same line number
    """
    # Load data
    human_data = load_jsonl(human_file)
    machine_data = load_jsonl(machine_file)
    
    # Check that files have the same number of lines
    if len(human_data) != len(machine_data):
        print(f"Warning: Files have different lengths:")
        print(f"  Human: {len(human_data)} lines")
        print(f"  Machine: {len(machine_data)} lines")
        print(f"  Will compare only the first {min(len(human_data), len(machine_data))} lines")
    
    # Match by line number and validate IDs match
    human_annotations = []
    llm_annotations = []
    n_compared = 0
    
    for line_num, (human_sample, machine_sample) in enumerate(zip(human_data, machine_data), start=1):
        human_id = human_sample.get('id')
        machine_id = machine_sample.get('id')
        
        # Validate IDs match
        if human_id != machine_id:
            raise ValueError(
                f"ID mismatch at line {line_num}:\n"
                f"  Human file ({human_file.name}): id = '{human_id}'\n"
                f"  Machine file ({machine_file.name}): id = '{machine_id}'\n"
                f"Files must have samples in the same order with matching IDs."
            )
        
        # Extract annotations
        human_cats = extract_human_annotations(human_sample)
        llm_cats = extract_llm_annotations(machine_sample)
        
        human_annotations.append(human_cats)
        llm_annotations.append(llm_cats)
        n_compared += 1
    
    # Calculate metrics
    metrics = calculate_metrics(human_annotations, llm_annotations, ALL_CATEGORIES)
    
    return {
        'machine_file': machine_file.name,
        'n_compared_samples': n_compared,
        'n_human_samples': len(human_data),
        'n_machine_samples': len(machine_data),
        **metrics
    }


def main():
    """Main function to compare human and machine annotations."""
    # Define paths
    script_dir = Path(__file__).parent
    human_dir = script_dir / 'results' / 'human_annotation'
    machine_dir = script_dir / 'results' / 'annotate'
    
    # Find human annotation file (hagenkortjuharova_sparc_annotated.jsonl)
    human_files = list(human_dir.glob('hagenkort*.jsonl'))
    
    if not human_files:
        print(f"Error: No human annotation file found in {human_dir}")
        return
    
    human_file = human_files[0]
    print(f"Human annotation file: {human_file.name}")
    print(f"=" * 80)
    
    # Find all machine annotation files starting with "annotation_samples"
    machine_files = list(machine_dir.glob('annotation_samples*.jsonl'))
    
    if not machine_files:
        print(f"Error: No machine annotation files found with prefix 'annotation_samples' in {machine_dir}")
        return
    
    print(f"Found {len(machine_files)} machine annotation file(s) with 'annotation_samples' prefix:\n")
    
    # Compare each machine annotation file with human annotations
    all_results = []
    
    for machine_file in sorted(machine_files):
        print(f"\n{'=' * 80}")
        print(f"Comparing with: {machine_file.name}")
        print(f"{'=' * 80}")
        
        results = compare_annotations(human_file, machine_file)
        
        if results:
            all_results.append(results)
            
            # Print results
            print(f"\nSamples:")
            print(f"  Human annotations: {results['n_human_samples']}")
            print(f"  Machine annotations: {results['n_machine_samples']}")
            print(f"  Compared samples: {results['n_compared_samples']}")
            
            print(f"\nMacro-averaged metrics:")
            print(f"  F1 Score:  {results['macro_f1']:.4f}")
            print(f"  Precision: {results['macro_precision']:.4f}")
            print(f"  Recall:    {results['macro_recall']:.4f}")
            
            print(f"\nPer-category metrics:")
            print(f"  {'Category':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
            print(f"  {'-' * 54}")
            
            for cat in ALL_CATEGORIES:
                cat_metrics = results['per_category'][cat]
                print(f"  {cat:<10} {cat_metrics['f1']:<8.4f} "
                      f"{cat_metrics['precision']:<10.4f} "
                      f"{cat_metrics['recall']:<8.4f} "
                      f"{cat_metrics['support']:<8}")
    
    # Summary table
    if len(all_results) > 1:
        print(f"\n\n{'=' * 80}")
        print("SUMMARY TABLE")
        print(f"{'=' * 80}")
        print(f"{'Model':<50} {'Macro F1':<12} {'Precision':<12} {'Recall':<10}")
        print(f"{'-' * 84}")
        
        for result in all_results:
            model_name = result['machine_file'].replace('annotation_samples.annotated_by_', '').replace('.jsonl', '')
            print(f"{model_name:<50} {result['macro_f1']:<12.4f} "
                  f"{result['macro_precision']:<12.4f} "
                  f"{result['macro_recall']:<10.4f}")
    
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()

