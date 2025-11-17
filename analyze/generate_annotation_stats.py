#!/usr/bin/env python3
"""
Generate LaTeX tables with key statistics about human annotations.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

# Mapping of failure reason codes to human-readable names
FAILURE_REASONS = {
    'a_planning_logical_flaw': 'A: Planning/Logical Flaw',
    'b_misunderstood_invented_rule': 'B: Misunderstood Rule',
    'c_spatial_geometric_misjudgment': 'C: Spatial Misjudgment',
    'd_premature_verification': 'D: Premature Verification',
    'e_no_correction_despite_noticing': 'E: No Correction',
    'f_grid_coordinate_error': 'F: Coordinate Error'
}

FAILURE_CODES = {
    'a_planning_logical_flaw': 'A',
    'b_misunderstood_invented_rule': 'B',
    'c_spatial_geometric_misjudgment': 'C',
    'd_premature_verification': 'D',
    'e_no_correction_despite_noticing': 'E',
    'f_grid_coordinate_error': 'F'
}


def load_annotations(filepath: Path) -> List[dict]:
    """Load annotations from a JSONL file."""
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def get_annotator_name(filepath: Path) -> str:
    """Extract annotator name from filepath."""
    name = filepath.stem.split('_')[0]
    return name.capitalize()


def calculate_basic_stats(annotations: List[dict]) -> Dict:
    """Calculate basic statistics for an annotator."""
    stats = {
        'total_samples': len(annotations),
        'solved': 0,
        'failed': 0,
        'annotated': 0,
        'failure_counts': defaultdict(int),
        'difficulty_distribution': defaultdict(int),
        'failure_per_sample': [],
        'co_occurrence': defaultdict(int)
    }
    
    for sample in annotations:
        # Difficulty distribution
        if 'difficulty_level' in sample:
            stats['difficulty_distribution'][sample['difficulty_level']] += 1
        
        # Solved status
        if sample.get('result', {}).get('solved', False):
            stats['solved'] += 1
        else:
            stats['failed'] += 1
        
        # Failure annotations
        failure_ann = sample.get('failure_annotation', {})
        if failure_ann and failure_ann.get('completed', False):
            stats['annotated'] += 1
            failure_reasons = failure_ann.get('failure_reasons', [])
            stats['failure_per_sample'].append(len(failure_reasons))
            
            # Count individual failures
            for reason in failure_reasons:
                stats['failure_counts'][reason] += 1
            
            # Count co-occurrences (pairs of failure types)
            for i, reason1 in enumerate(failure_reasons):
                for reason2 in failure_reasons[i+1:]:
                    pair = tuple(sorted([FAILURE_CODES[reason1], FAILURE_CODES[reason2]]))
                    stats['co_occurrence'][pair] += 1
    
    # Calculate average failures per sample
    if stats['failure_per_sample']:
        stats['avg_failures_per_sample'] = statistics.mean(stats['failure_per_sample'])
        stats['std_failures_per_sample'] = statistics.stdev(stats['failure_per_sample']) if len(stats['failure_per_sample']) > 1 else 0
    else:
        stats['avg_failures_per_sample'] = 0
        stats['std_failures_per_sample'] = 0
    
    return stats


def calculate_agreement(annotations_list: List[List[dict]]) -> Dict:
    """Calculate inter-annotator agreement metrics."""
    # Match samples by ID
    samples_by_id = defaultdict(list)
    for annotator_idx, annotations in enumerate(annotations_list):
        for sample in annotations:
            sample_id = sample.get('id')
            failure_ann = sample.get('failure_annotation', {})
            if failure_ann and failure_ann.get('completed', False):
                failure_reasons = set(failure_ann.get('failure_reasons', []))
                samples_by_id[sample_id].append((annotator_idx, failure_reasons))
    
    # Only consider samples annotated by all annotators
    common_samples = {sid: anns for sid, anns in samples_by_id.items() if len(anns) == len(annotations_list)}
    
    # Calculate pairwise F1 scores
    pairwise_f1 = []
    num_annotators = len(annotations_list)
    
    for i in range(num_annotators):
        for j in range(i+1, num_annotators):
            precisions = []
            recalls = []
            f1_scores = []
            
            for sample_id, annotations in common_samples.items():
                ann_dict = {idx: reasons for idx, reasons in annotations}
                reasons_i = ann_dict.get(i, set())
                reasons_j = ann_dict.get(j, set())
                
                if len(reasons_i) == 0 and len(reasons_j) == 0:
                    continue  # Skip if both annotators found no failures
                
                if len(reasons_i) == 0 or len(reasons_j) == 0:
                    f1_scores.append(0.0)
                    continue
                
                intersection = len(reasons_i & reasons_j)
                if intersection == 0:
                    f1_scores.append(0.0)
                else:
                    precision = intersection / len(reasons_j)
                    recall = intersection / len(reasons_i)
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)
                    precisions.append(precision)
                    recalls.append(recall)
            
            if f1_scores:
                pairwise_f1.append(statistics.mean(f1_scores))
    
    agreement_stats = {
        'common_samples': len(common_samples),
        'pairwise_f1': pairwise_f1,
        'avg_pairwise_f1': statistics.mean(pairwise_f1) if pairwise_f1 else 0.0,
        'std_pairwise_f1': statistics.stdev(pairwise_f1) if len(pairwise_f1) > 1 else 0.0
    }
    
    # Calculate category-wise agreement
    category_agreement = {}
    for category_code, category_name in FAILURE_REASONS.items():
        category_present = defaultdict(lambda: [False] * num_annotators)
        
        for sample_id, annotations in common_samples.items():
            for annotator_idx, reasons in annotations:
                category_present[sample_id][annotator_idx] = category_code in reasons
        
        # Calculate percentage agreement
        agreements = []
        for sample_id, presence in category_present.items():
            if sum(presence) == 0 or sum(presence) == num_annotators:
                agreements.append(1.0)  # All agree (all yes or all no)
            else:
                agreements.append(0.0)  # Disagree
        
        category_agreement[category_code] = {
            'agreement_rate': statistics.mean(agreements) if agreements else 0.0,
            'samples_with_category': sum(any(presence) for presence in category_present.values())
        }
    
    agreement_stats['category_agreement'] = category_agreement
    
    return agreement_stats


def generate_main_stats_table(annotator_stats: Dict[str, Dict], agreement_stats: Dict) -> str:
    """Generate the main statistics LaTeX table."""
    latex = []
    latex.append("% Main Statistics Table")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Human Annotation Dataset Statistics}")
    latex.append("\\label{tab:annotation_stats}")
    latex.append("\\begin{tabular}{lc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    latex.append("\\midrule")
    
    # Number of annotators
    num_annotators = len(annotator_stats)
    latex.append(f"Number of annotators & {num_annotators} \\\\")
    
    # Total samples per annotator
    annotator_names = sorted(annotator_stats.keys())
    total_samples = annotator_stats[annotator_names[0]]['total_samples']
    latex.append(f"Samples per annotator & {total_samples} \\\\")
    
    # Common annotated samples
    latex.append(f"Overlapping samples & {agreement_stats['common_samples']} \\\\")
    
    # Total unique samples (across all annotators)
    latex.append(f"Total annotations collected & {num_annotators * total_samples} \\\\")
    
    latex.append("\\midrule")
    
    # Average failures per sample
    values = [annotator_stats[name]['avg_failures_per_sample'] for name in annotator_names]
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0
    latex.append(f"Avg. failure types per sample & {mean_val:.2f} $\\pm$ {std_val:.2f} \\\\")
    
    # Range of failures per sample
    all_failures = []
    for stats in annotator_stats.values():
        all_failures.extend(stats['failure_per_sample'])
    if all_failures:
        min_fail = min(all_failures)
        max_fail = max(all_failures)
        latex.append(f"Failure types range & {min_fail}--{max_fail} \\\\")
    
    latex.append("\\midrule")
    
    # Inter-annotator agreement
    latex.append(f"Inter-annotator agreement (F1) & {agreement_stats['avg_pairwise_f1']:.3f} $\\pm$ {agreement_stats['std_pairwise_f1']:.3f} \\\\")
    
    # Pairwise F1 range
    if agreement_stats['pairwise_f1']:
        min_f1 = min(agreement_stats['pairwise_f1'])
        max_f1 = max(agreement_stats['pairwise_f1'])
        latex.append(f"Agreement range (F1) & {min_f1:.3f}--{max_f1:.3f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_failure_distribution_table(annotator_stats: Dict[str, Dict]) -> str:
    """Generate failure type distribution LaTeX table."""
    latex = []
    latex.append("\n% Failure Type Distribution Table")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Aggregated Failure Type Distribution}")
    latex.append("\\label{tab:failure_distribution}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Failure Type} & \\textbf{Count} & \\textbf{Prevalence} & \\textbf{Avg. per Sample} \\\\")
    latex.append("\\midrule")
    
    # Aggregate across all annotators
    total_annotations = sum(stats['annotated'] for stats in annotator_stats.values())
    
    # For each failure type
    for reason_code, reason_name in FAILURE_REASONS.items():
        # Total count across all annotators
        total_count = sum(stats['failure_counts'][reason_code] for stats in annotator_stats.values())
        
        # Prevalence: percentage of samples that have this failure type
        # (counting how many unique samples had this annotation)
        prevalence_pct = (total_count / total_annotations * 100) if total_annotations > 0 else 0
        
        # Average occurrences per annotated sample
        avg_per_sample = total_count / total_annotations if total_annotations > 0 else 0
        
        row = f"{reason_name} & {total_count} & {prevalence_pct:.1f}\\% & {avg_per_sample:.2f} \\\\"
        latex.append(row)
    
    latex.append("\\midrule")
    
    # Total row
    grand_total = sum(sum(stats['failure_counts'].values()) for stats in annotator_stats.values())
    avg_total = grand_total / total_annotations if total_annotations > 0 else 0
    latex.append(f"\\textbf{{Total}} & {grand_total} & --- & {avg_total:.2f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_difficulty_table(annotator_stats: Dict[str, Dict]) -> str:
    """Generate difficulty distribution LaTeX table."""
    latex = []
    latex.append("\n% Difficulty Distribution Table")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Sample Distribution by Difficulty Level}")
    latex.append("\\label{tab:difficulty_distribution}")
    latex.append("\\begin{tabular}{ccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Difficulty Level} & \\textbf{Samples} & \\textbf{Percentage} \\\\")
    latex.append("\\midrule")
    
    # Get all difficulty levels and aggregate counts
    difficulty_counts = defaultdict(int)
    for stats in annotator_stats.values():
        for level, count in stats['difficulty_distribution'].items():
            difficulty_counts[level] += count
    
    # Calculate total for percentages
    total_samples = sum(difficulty_counts.values())
    
    # Sort by difficulty level
    for level in sorted(difficulty_counts.keys()):
        count = difficulty_counts[level]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        latex.append(f"{level} & {count} & {percentage:.1f}\\% \\\\")
    
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Total}} & {total_samples} & 100.0\\% \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_co_occurrence_table(annotator_stats: Dict[str, Dict]) -> str:
    """Generate failure type co-occurrence matrix."""
    latex = []
    latex.append("\n% Failure Co-occurrence Table")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Failure Type Co-occurrence Matrix}")
    latex.append("\\label{tab:failure_cooccurrence}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    
    # Aggregate co-occurrences across all annotators
    total_cooccurrence = defaultdict(int)
    for stats in annotator_stats.values():
        for pair, count in stats['co_occurrence'].items():
            total_cooccurrence[pair] += count
    
    codes = ['A', 'B', 'C', 'D', 'E', 'F']
    header = "& " + " & ".join([f"\\textbf{{{c}}}" for c in codes]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    for i, code1 in enumerate(codes):
        row_vals = []
        for j, code2 in enumerate(codes):
            if i >= j:
                row_vals.append("---")
            else:
                pair = tuple(sorted([code1, code2]))
                count = total_cooccurrence.get(pair, 0)
                row_vals.append(str(count) if count > 0 else "---")
        
        row = f"\\textbf{{{code1}}} & " + " & ".join(row_vals) + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_category_agreement_table(agreement_stats: Dict) -> str:
    """Generate per-category agreement table."""
    latex = []
    latex.append("\n% Category Agreement Table")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Per-Category Agreement Rates}")
    latex.append("\\label{tab:category_agreement}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Failure Type} & \\textbf{Agreement Rate} & \\textbf{Prevalence} \\\\")
    latex.append("\\midrule")
    
    for reason_code, reason_name in FAILURE_REASONS.items():
        cat_stats = agreement_stats['category_agreement'].get(reason_code, {})
        agreement_rate = cat_stats.get('agreement_rate', 0) * 100
        prevalence = cat_stats.get('samples_with_category', 0)
        
        row = f"{reason_name} & {agreement_rate:.1f}\\% & {prevalence} \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for annotation statistics')
    parser.add_argument('--input-dir', type=Path, 
                       default=Path('analyze/results/human_annotation'),
                       help='Directory containing annotation files')
    parser.add_argument('--output', type=Path,
                       default=Path('analyze/results/annotation_stats_tables.tex'),
                       help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Find all annotation files
    annotation_files = [
        args.input_dir / 'abukhanov_sparc_annotated.jsonl',
        args.input_dir / 'hagenkort_sparc_annotated.jsonl',
        args.input_dir / 'juharova_sparc_annotated.jsonl'
    ]
    
    # Load annotations
    print("Loading annotations...")
    all_annotations = []
    annotator_stats = {}
    
    for filepath in annotation_files:
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"  Processing {filepath.name}...")
        annotations = load_annotations(filepath)
        all_annotations.append(annotations)
        
        annotator_name = get_annotator_name(filepath)
        stats = calculate_basic_stats(annotations)
        annotator_stats[annotator_name] = stats
        
        print(f"    Total samples: {stats['total_samples']}")
        print(f"    Annotated: {stats['annotated']}")
    
    # Calculate agreement
    print("\nCalculating inter-annotator agreement...")
    agreement_stats = calculate_agreement(all_annotations)
    print(f"  Common samples: {agreement_stats['common_samples']}")
    print(f"  Average pairwise F1: {agreement_stats['avg_pairwise_f1']:.3f}")
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    latex_output = []
    
    # Add header
    latex_output.append("% LaTeX tables for human annotation statistics")
    latex_output.append("% Generated automatically by generate_annotation_stats.py")
    latex_output.append("% Requires \\usepackage{booktabs} in your LaTeX document\n")
    
    # Generate all tables
    latex_output.append(generate_main_stats_table(annotator_stats, agreement_stats))
    latex_output.append(generate_failure_distribution_table(annotator_stats))
    latex_output.append(generate_difficulty_table(annotator_stats))
    latex_output.append(generate_co_occurrence_table(annotator_stats))
    latex_output.append(generate_category_agreement_table(agreement_stats))
    
    # Save output
    output_content = "\n\n".join(latex_output)
    args.output.write_text(output_content)
    print(f"\n✓ LaTeX tables saved to: {args.output}")
    
    # Print summary statistics to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for name in sorted(annotator_stats.keys()):
        stats = annotator_stats[name]
        print(f"\n{name}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Annotated: {stats['annotated']} ({stats['annotated']/max(stats['failed'],1)*100:.1f}%)")
        print(f"  Avg failures per sample: {stats['avg_failures_per_sample']:.2f} ± {stats['std_failures_per_sample']:.2f}")
        print(f"  Failure distribution:")
        for reason_code, count in sorted(stats['failure_counts'].items()):
            pct = count / stats['annotated'] * 100 if stats['annotated'] > 0 else 0
            print(f"    {FAILURE_CODES[reason_code]}: {count} ({pct:.1f}%)")
    
    print(f"\n\nInter-annotator Agreement:")
    print(f"  Common samples: {agreement_stats['common_samples']}")
    print(f"  Pairwise F1 scores: {', '.join([f'{f1:.3f}' for f1 in agreement_stats['pairwise_f1']])}")
    print(f"  Average: {agreement_stats['avg_pairwise_f1']:.3f} ± {agreement_stats['std_pairwise_f1']:.3f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

