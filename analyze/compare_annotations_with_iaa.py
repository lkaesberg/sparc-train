#!/usr/bin/env python3
"""
Compare human annotations with machine (LLM) annotations including inter-annotator agreement.

This script:
1. Calculates inter-annotator agreement (IAA) between human annotators
2. Creates a majority-vote baseline (annotations appearing in at least 2/3 annotators)
3. Compares LLM annotations against individual annotators and majority vote
4. Calculates macro F1, precision, recall, and Hamming loss across all 6 failure categories (A-F)

Failure categories:
  A: Planning/logical flaw in the reasoning approach
  B: Misunderstood or invented puzzle rules
  C: Spatial/geometric misjudgment or miscalculation
  D: Premature verification - claims correctness without checking key rules
  E: No correction despite noticing - recognizes errors but doesn't adjust
  F: Grid/coordinate error - off-by-one, swapped x/y, or out-of-bounds steps

Usage:
    python analyze/compare_annotations_with_iaa.py
    python analyze/compare_annotations_with_iaa.py --exclude-categories A
    python analyze/compare_annotations_with_iaa.py --exclude-categories A,C
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from itertools import combinations
import statistics
import argparse


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


def filter_annotations(annotations: List[Set[str]], categories_to_keep: List[str]) -> List[Set[str]]:
    """
    Filter annotations to only include specified categories.
    
    Args:
        annotations: List of annotation sets
        categories_to_keep: List of category codes to keep
    
    Returns:
        Filtered list of annotation sets
    """
    categories_set = set(categories_to_keep)
    return [cats & categories_set for cats in annotations]


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


def calculate_hamming_loss(human_annotations: List[Set[str]], 
                           llm_annotations: List[Set[str]],
                           all_categories: List[str]) -> float:
    """
    Calculate Hamming loss for multi-label classification.
    
    Hamming loss is the fraction of labels that are incorrectly predicted.
    Lower is better (0 = perfect, 1 = all wrong).
    
    Args:
        human_annotations: List of sets containing human annotation categories
        llm_annotations: List of sets containing LLM annotation categories
        all_categories: List of all possible category codes
    
    Returns:
        Hamming loss value (0 to 1)
    """
    n_samples = len(human_annotations)
    n_labels = len(all_categories)
    
    if n_samples == 0 or n_labels == 0:
        return 0.0
    
    total_incorrect = 0
    for human_cats, llm_cats in zip(human_annotations, llm_annotations):
        # For each sample, count incorrect predictions
        for category in all_categories:
            human_has = category in human_cats
            llm_has = category in llm_cats
            # XOR: incorrect if they differ
            if human_has != llm_has:
                total_incorrect += 1
    
    # Hamming loss = (incorrect labels) / (total samples * total labels)
    hamming_loss = total_incorrect / (n_samples * n_labels)
    return hamming_loss


def calculate_metrics(human_annotations: List[Set[str]], 
                      llm_annotations: List[Set[str]],
                      all_categories: List[str]) -> Dict:
    """
    Calculate macro F1, precision, recall, and Hamming loss.
    
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
    
    # Calculate Hamming loss
    hamming_loss = calculate_hamming_loss(human_annotations, llm_annotations, all_categories)
    
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'hamming_loss': hamming_loss,
        'per_category': per_category_metrics,
        'n_samples': n_samples
    }


def calculate_pairwise_agreement(annotations1: List[Set[str]], 
                                  annotations2: List[Set[str]],
                                  all_categories: List[str]) -> Dict:
    """
    Calculate pairwise agreement between two annotators.
    
    Returns:
        Dictionary with F1 score and per-category metrics
    """
    return calculate_metrics(annotations1, annotations2, all_categories)


def create_majority_vote(all_human_annotations: List[List[Set[str]]], 
                         min_votes: int = 2) -> List[Set[str]]:
    """
    Create majority-vote annotations where a category is included if 
    at least min_votes annotators selected it.
    
    Args:
        all_human_annotations: List of [annotator1_annotations, annotator2_annotations, ...]
        min_votes: Minimum number of annotators needed to include a category
    
    Returns:
        List of majority-vote annotation sets
    """
    n_samples = len(all_human_annotations[0])
    majority_annotations = []
    
    for sample_idx in range(n_samples):
        # Count votes for each category
        category_votes = {cat: 0 for cat in ALL_CATEGORIES}
        
        for annotator_annotations in all_human_annotations:
            sample_cats = annotator_annotations[sample_idx]
            for cat in sample_cats:
                category_votes[cat] += 1
        
        # Include categories with at least min_votes
        majority_cats = {cat for cat, votes in category_votes.items() if votes >= min_votes}
        majority_annotations.append(majority_cats)
    
    return majority_annotations


def calculate_fleiss_kappa_per_category(all_human_annotations: List[List[Set[str]]], 
                                         category: str) -> float:
    """
    Calculate Fleiss' Kappa for a specific category across all annotators.
    
    Args:
        all_human_annotations: List of [annotator1_annotations, annotator2_annotations, ...]
        category: The category to calculate kappa for
    
    Returns:
        Fleiss' Kappa value
    """
    n_samples = len(all_human_annotations[0])
    n_annotators = len(all_human_annotations)
    
    # Create binary matrix: rows = samples, cols = annotators
    # 1 if category is present, 0 otherwise
    votes = []
    for sample_idx in range(n_samples):
        sample_votes = [1 if category in all_human_annotations[annotator_idx][sample_idx] 
                       else 0 
                       for annotator_idx in range(n_annotators)]
        votes.append(sample_votes)
    
    # Calculate Fleiss' Kappa
    # P_i: proportion of all annotator pairs that agree for item i
    p_i_values = []
    for sample_votes in votes:
        n_positive = sum(sample_votes)
        n_negative = n_annotators - n_positive
        # Number of agreeing pairs / total pairs
        agreeing_pairs = (n_positive * (n_positive - 1) + n_negative * (n_negative - 1)) / 2
        total_pairs = (n_annotators * (n_annotators - 1)) / 2
        p_i = agreeing_pairs / total_pairs if total_pairs > 0 else 0
        p_i_values.append(p_i)
    
    # P_bar: mean of P_i across all items
    p_bar = sum(p_i_values) / n_samples if n_samples > 0 else 0
    
    # P_e: expected agreement by chance
    # Calculate proportion of positive and negative annotations
    total_annotations = n_samples * n_annotators
    n_positive_total = sum(sum(sample_votes) for sample_votes in votes)
    n_negative_total = total_annotations - n_positive_total
    
    p_positive = n_positive_total / total_annotations if total_annotations > 0 else 0
    p_negative = n_negative_total / total_annotations if total_annotations > 0 else 0
    
    p_e = p_positive ** 2 + p_negative ** 2
    
    # Fleiss' Kappa
    kappa = (p_bar - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0
    
    return kappa


def main():
    """Main function to compare human and machine annotations with IAA."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compare human and machine annotations with inter-annotator agreement analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--exclude-categories',
        type=str,
        default='',
        help='Comma-separated list of categories to exclude (e.g., "A,C" to exclude categories A and C)'
    )
    args = parser.parse_args()
    
    # Parse excluded categories
    excluded_categories = set()
    if args.exclude_categories:
        excluded_categories = set(cat.strip().upper() for cat in args.exclude_categories.split(','))
        # Validate categories
        invalid_cats = excluded_categories - set(ALL_CATEGORIES)
        if invalid_cats:
            print(f"Error: Invalid categories: {', '.join(invalid_cats)}")
            print(f"Valid categories are: {', '.join(ALL_CATEGORIES)}")
            return
    
    # Filter categories to analyze
    categories_to_analyze = [cat for cat in ALL_CATEGORIES if cat not in excluded_categories]
    
    if not categories_to_analyze:
        print("Error: Cannot exclude all categories!")
        return
    
    # Define paths
    script_dir = Path(__file__).parent
    human_dir = script_dir / 'results' / 'human_annotation'
    machine_dir = script_dir / 'results' / 'annotate'
    
    # Find all human annotation files
    human_files = sorted([
        human_dir / 'abukhanov_sparc_annotated.jsonl',
        human_dir / 'hagenkort_sparc_annotated.jsonl',
        human_dir / 'juharova_sparc_annotated.jsonl',
    ])
    
    # Check that all files exist
    for f in human_files:
        if not f.exists():
            print(f"Error: Human annotation file not found: {f}")
            return
    
    print(f"{'=' * 80}")
    print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print(f"{'=' * 80}")
    
    # Display category filter information
    if excluded_categories:
        print(f"\n⚠️  FILTERING: Excluding categories: {', '.join(sorted(excluded_categories))}")
        print(f"    Analyzing categories: {', '.join(categories_to_analyze)}")
    else:
        print(f"\n    Analyzing all categories: {', '.join(categories_to_analyze)}")
    
    print(f"\nLoading {len(human_files)} human annotation files:")
    for f in human_files:
        print(f"  - {f.name}")
    
    # Load all human annotations
    all_human_data = [load_jsonl(f) for f in human_files]
    
    # Verify all files have the same length and IDs match
    n_samples = len(all_human_data[0])
    for i, data in enumerate(all_human_data[1:], start=1):
        if len(data) != n_samples:
            print(f"\nError: File {human_files[i].name} has {len(data)} samples, expected {n_samples}")
            return
        
        # Check IDs match at each line
        for line_num, (sample1, sample2) in enumerate(zip(all_human_data[0], data), start=1):
            if sample1.get('id') != sample2.get('id'):
                print(f"\nError: ID mismatch at line {line_num} between:")
                print(f"  {human_files[0].name}: {sample1.get('id')}")
                print(f"  {human_files[i].name}: {sample2.get('id')}")
                return
    
    print(f"\n✓ All files have {n_samples} samples with matching IDs")
    
    # Extract annotations from all human annotators
    all_human_annotations = []
    for human_data in all_human_data:
        annotations = [extract_human_annotations(sample) for sample in human_data]
        # Filter to only include categories we're analyzing
        filtered_annotations = filter_annotations(annotations, categories_to_analyze)
        all_human_annotations.append(filtered_annotations)
    
    # Calculate pairwise agreement between human annotators
    print(f"\n{'=' * 80}")
    print("PAIRWISE INTER-ANNOTATOR AGREEMENT (Human vs Human)")
    print(f"{'=' * 80}")
    
    annotator_names = ['Abukhanov', 'Hagenkort', 'Juharova']
    pairwise_f1_scores = []
    
    for (i, name1), (j, name2) in combinations(enumerate(annotator_names), 2):
        metrics = calculate_pairwise_agreement(
            all_human_annotations[i], 
            all_human_annotations[j],
            categories_to_analyze
        )
        pairwise_f1_scores.append(metrics['macro_f1'])
        
        print(f"\n{name1} vs {name2}:")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Per-category F1:")
        for cat in categories_to_analyze:
            cat_f1 = metrics['per_category'][cat]['f1']
            print(f"    {cat}: {cat_f1:.4f}")
    
    avg_pairwise_f1 = statistics.mean(pairwise_f1_scores)
    print(f"\n{'=' * 80}")
    print(f"Average Pairwise F1 Score: {avg_pairwise_f1:.4f}")
    print(f"{'=' * 80}")
    
    # Calculate Fleiss' Kappa for each category
    print(f"\n{'=' * 80}")
    print("FLEISS' KAPPA (Multi-Rater Agreement)")
    print(f"{'=' * 80}")
    print(f"\nPer-category Fleiss' Kappa:")
    
    kappa_scores = []
    for cat in categories_to_analyze:
        kappa = calculate_fleiss_kappa_per_category(all_human_annotations, cat)
        kappa_scores.append(kappa)
        
        # Interpretation
        if kappa < 0:
            interpretation = "Poor (less than chance)"
        elif kappa < 0.20:
            interpretation = "Slight"
        elif kappa < 0.40:
            interpretation = "Fair"
        elif kappa < 0.60:
            interpretation = "Moderate"
        elif kappa < 0.80:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"
        
        print(f"  {cat}: {kappa:.4f} ({interpretation})")
    
    avg_kappa = statistics.mean(kappa_scores)
    print(f"\nAverage Fleiss' Kappa: {avg_kappa:.4f}")
    
    # Create majority-vote baseline (at least 2/3 annotators)
    print(f"\n{'=' * 80}")
    print("MAJORITY VOTE BASELINE (≥2/3 annotators)")
    print(f"{'=' * 80}")
    
    majority_annotations = create_majority_vote(all_human_annotations, min_votes=2)
    
    # Show distribution of majority annotations
    majority_category_counts = {cat: 0 for cat in categories_to_analyze}
    for sample_cats in majority_annotations:
        for cat in sample_cats:
            if cat in categories_to_analyze:
                majority_category_counts[cat] += 1
    
    print(f"\nMajority-vote annotation counts:")
    for cat in categories_to_analyze:
        print(f"  {cat}: {majority_category_counts[cat]} samples")
    
    # Find machine annotation files
    machine_files = list(machine_dir.glob('annotation_samples*.jsonl'))
    
    if not machine_files:
        print(f"\nNo machine annotation files found with prefix 'annotation_samples'")
        return
    
    print(f"\n{'=' * 80}")
    print("LLM vs HUMAN ANNOTATORS COMPARISON")
    print(f"{'=' * 80}")
    print(f"\nFound {len(machine_files)} machine annotation file(s)\n")
    
    # Store results for summary table
    all_model_results = []
    
    # Compare LLM against each human annotator and majority vote
    for machine_file in sorted(machine_files):
        machine_data = load_jsonl(machine_file)
        
        if len(machine_data) != n_samples:
            print(f"\nWarning: {machine_file.name} has {len(machine_data)} samples, expected {n_samples}")
            continue
        
        # Verify IDs match
        try:
            for line_num, (human_sample, machine_sample) in enumerate(zip(all_human_data[0], machine_data), start=1):
                if human_sample.get('id') != machine_sample.get('id'):
                    raise ValueError(
                        f"ID mismatch at line {line_num}:\n"
                        f"  Human: {human_sample.get('id')}\n"
                        f"  Machine ({machine_file.name}): {machine_sample.get('id')}"
                    )
        except ValueError as e:
            print(f"\nError: {e}")
            continue
        
        # Extract LLM annotations and filter to categories being analyzed
        llm_annotations = [extract_llm_annotations(sample) for sample in machine_data]
        llm_annotations_filtered = filter_annotations(llm_annotations, categories_to_analyze)
        
        model_name = machine_file.name.replace('annotation_samples.annotated_by_', '').replace('.jsonl', '')
        if model_name == 'annotation_samples.annotated':
            model_name = 'annotation_samples'
        
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")
        
        # Compare against each individual annotator
        print(f"\nComparison with individual annotators:")
        individual_f1_scores = []
        
        for i, name in enumerate(annotator_names):
            metrics = calculate_metrics(all_human_annotations[i], llm_annotations_filtered, categories_to_analyze)
            individual_f1_scores.append(metrics['macro_f1'])
            print(f"  {name}: Macro F1 = {metrics['macro_f1']:.4f}")
        
        avg_individual_f1 = statistics.mean(individual_f1_scores)
        print(f"  Average: Macro F1 = {avg_individual_f1:.4f}")
        
        # Compare against majority vote
        print(f"\nComparison with majority vote (≥2/3):")
        majority_metrics = calculate_metrics(majority_annotations, llm_annotations_filtered, categories_to_analyze)
        print(f"  Macro F1:      {majority_metrics['macro_f1']:.4f}")
        print(f"  Precision:     {majority_metrics['macro_precision']:.4f}")
        print(f"  Recall:        {majority_metrics['macro_recall']:.4f}")
        print(f"  Hamming Loss:  {majority_metrics['hamming_loss']:.4f} (lower is better)")
        
        print(f"\n  Per-category metrics vs majority vote:")
        print(f"  {'Category':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
        print(f"  {'-' * 54}")
        
        for cat in categories_to_analyze:
            cat_metrics = majority_metrics['per_category'][cat]
            print(f"  {cat:<10} {cat_metrics['f1']:<8.4f} "
                  f"{cat_metrics['precision']:<10.4f} "
                  f"{cat_metrics['recall']:<8.4f} "
                  f"{cat_metrics['support']:<8}")
        
        # Store results for summary table
        all_model_results.append({
            'model': model_name,
            'majority_f1': majority_metrics['macro_f1'],
            'majority_precision': majority_metrics['macro_precision'],
            'majority_recall': majority_metrics['macro_recall'],
            'hamming_loss': majority_metrics['hamming_loss'],
            'avg_individual_f1': avg_individual_f1,
            'per_category_f1': {cat: majority_metrics['per_category'][cat]['f1'] for cat in categories_to_analyze}
        })
    
    # Sort models by F1 score (descending)
    all_model_results.sort(key=lambda x: x['majority_f1'], reverse=True)
    
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON TABLE (vs Majority Vote ≥2/3)")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<45} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Hamming':<10}")
    print(f"{'-' * 87}")
    
    for result in all_model_results:
        print(f"{result['model']:<45} {result['majority_f1']:<10.4f} "
              f"{result['majority_precision']:<12.4f} {result['majority_recall']:<10.4f} "
              f"{result['hamming_loss']:<10.4f}")
    
    # Add human baseline
    print(f"{'-' * 87}")
    print(f"{'Human Baseline (Avg Pairwise)':<45} {avg_pairwise_f1:<10.4f} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
    print(f"\nNote: Hamming Loss = fraction of incorrectly predicted labels (lower is better, 0 = perfect)")
    
    # Per-category comparison table
    print(f"\n{'=' * 80}")
    print("PER-CATEGORY F1 SCORES (vs Majority Vote ≥2/3)")
    print(f"{'=' * 80}")
    
    # Build header dynamically based on categories being analyzed
    header = f"\n{'Model':<45} " + ' '.join([f"{cat:<8}" for cat in categories_to_analyze])
    print(header)
    separator_length = 45 + 8 * len(categories_to_analyze) + len(categories_to_analyze) - 1
    print(f"{'-' * separator_length}")
    
    for result in all_model_results:
        cat_scores = ' '.join([f"{result['per_category_f1'][cat]:<8.4f}" for cat in categories_to_analyze])
        print(f"{result['model']:<45} {cat_scores}")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nInter-Annotator Agreement:")
    print(f"  Average Pairwise F1: {avg_pairwise_f1:.4f}")
    print(f"  Average Fleiss' Kappa: {avg_kappa:.4f}")
    print(f"\nInterpretation:")
    if avg_pairwise_f1 >= 0.70:
        print(f"  ✓ Strong agreement between human annotators (F1 ≥ 0.70)")
    elif avg_pairwise_f1 >= 0.50:
        print(f"  ⚠ Moderate agreement between human annotators (0.50 ≤ F1 < 0.70)")
    else:
        print(f"  ⚠ Low agreement between human annotators (F1 < 0.50)")
    
    # Highlight best models
    if all_model_results:
        print(f"\nBest Performing Models (vs Majority Vote):")
        for i, result in enumerate(all_model_results[:3], 1):
            comparison = "above" if result['majority_f1'] > avg_pairwise_f1 else "below"
            print(f"  {i}. {result['model']}: F1 = {result['majority_f1']:.4f} ({comparison} human baseline)")
    
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()

