#!/usr/bin/env python3
"""
Sample N random entries from all JSONL result files in analyze/results/sparc
with an even distribution across difficulty levels (1-5) and no duplicates.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def load_jsonl_files(results_dir):
    """Load all JSONL files from the results directory."""
    results_path = Path(results_dir)
    all_entries = []
    
    # Find all .jsonl files
    jsonl_files = list(results_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for jsonl_file in jsonl_files:
        # Track IDs within this file to avoid duplicates from the same file
        seen_ids_in_file = set()
        entries_from_file = 0
        duplicates_in_file = 0
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entry_id = entry.get('id')
                    
                    # Skip duplicates within the same file
                    if entry_id and entry_id in seen_ids_in_file:
                        duplicates_in_file += 1
                        continue
                    
                    if entry_id:
                        seen_ids_in_file.add(entry_id)
                    
                    # Store the source file with the entry
                    entry['_source_file'] = jsonl_file.name
                    all_entries.append(entry)
                    entries_from_file += 1
        
        if duplicates_in_file > 0:
            print(f"  {jsonl_file.name}: {entries_from_file} entries ({duplicates_in_file} duplicates removed)")
        else:
            print(f"  {jsonl_file.name}: {entries_from_file} entries")
    
    print(f"Loaded {len(all_entries)} total entries")
    return all_entries


def filter_by_solved_status(entries, solved_filter):
    """
    Filter entries by solved status.
    
    Args:
        entries: List of entries
        solved_filter: 'solved', 'unsolved', or None (no filtering)
    
    Returns:
        Filtered list of entries
    """
    if solved_filter is None:
        return entries
    
    filtered = []
    for entry in entries:
        result = entry.get('result', {})
        is_solved = result.get('solved', False)
        
        if solved_filter == 'solved' and is_solved:
            filtered.append(entry)
        elif solved_filter == 'unsolved' and not is_solved:
            filtered.append(entry)
    
    return filtered


def group_by_difficulty(entries):
    """Group entries by difficulty level."""
    difficulty_groups = defaultdict(list)
    
    for entry in entries:
        difficulty = entry.get('difficulty_level')
        if difficulty is not None:
            difficulty_groups[difficulty].append(entry)
    
    return difficulty_groups


def sample_evenly(difficulty_groups, n_samples):
    """
    Sample N entries with even distribution across difficulty levels.
    
    Args:
        difficulty_groups: Dict mapping difficulty level to list of entries
        n_samples: Total number of samples to draw
    
    Returns:
        List of sampled entries
    """
    # Get all difficulty levels (1-5)
    difficulty_levels = sorted(difficulty_groups.keys())
    
    if not difficulty_levels:
        print("No entries found with difficulty levels")
        return []
    
    print(f"\nDifficulty level distribution:")
    for level in difficulty_levels:
        print(f"  Level {level}: {len(difficulty_groups[level])} entries")
    
    # Calculate samples per difficulty level
    samples_per_level = n_samples // len(difficulty_levels)
    remaining_samples = n_samples % len(difficulty_levels)
    
    print(f"\nSampling strategy:")
    print(f"  {samples_per_level} samples per level")
    print(f"  {remaining_samples} additional samples to distribute")
    
    sampled_entries = []
    
    # Sample evenly from each difficulty level
    for level in difficulty_levels:
        available = difficulty_groups[level]
        # Base samples for this level
        n_to_sample = samples_per_level
        
        # Add one extra sample to first N levels if there are remaining samples
        if remaining_samples > 0:
            n_to_sample += 1
            remaining_samples -= 1
        
        # Don't try to sample more than available
        n_to_sample = min(n_to_sample, len(available))
        
        # Random sample without replacement
        samples = random.sample(available, n_to_sample)
        sampled_entries.extend(samples)
        
        print(f"  Level {level}: sampled {n_to_sample} entries")
    
    # Shuffle the final list to mix difficulty levels
    random.shuffle(sampled_entries)
    
    return sampled_entries


def save_samples(samples, output_file):
    """Save sampled entries to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for entry in samples:
            # Remove the internal _source_file field before saving
            entry_copy = entry.copy()
            entry_copy.pop('_source_file', None)
            f.write(json.dumps(entry_copy) + '\n')
    
    print(f"\nSaved {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample N random entries from JSONL files with even difficulty distribution"
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        required=True,
        help='Number of samples to draw'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='analyze/results/sparc',
        help='Directory containing JSONL result files (default: analyze/results/sparc)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='analyze/results/sampled_results.jsonl',
        help='Output file path (default: analyze/results/sampled_results.jsonl)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '-f', '--filter',
        type=str,
        choices=['solved', 'unsolved'],
        default=None,
        help='Filter samples by solved status: "solved" or "unsolved" (optional)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Load all entries (duplicates within same file already removed)
    print(f"Loading JSONL files from {args.input_dir}...")
    all_entries = load_jsonl_files(args.input_dir)
    
    print(f"\nNote: Same puzzle ID can appear multiple times across different files")
    
    # Filter by solved status if requested
    if args.filter:
        print(f"\nFiltering for {args.filter} samples...")
        filtered_entries = filter_by_solved_status(all_entries, args.filter)
        print(f"After filtering: {len(filtered_entries)} {args.filter} entries (from {len(all_entries)} total)")
        all_entries = filtered_entries
    
    # Group by difficulty
    difficulty_groups = group_by_difficulty(all_entries)
    
    # Sample evenly
    samples = sample_evenly(difficulty_groups, args.num_samples)
    
    # Save to output file
    save_samples(samples, args.output)
    
    # Print final statistics
    print(f"\nFinal sample statistics:")
    final_difficulty_counts = defaultdict(int)
    solved_count = 0
    unsolved_count = 0
    
    for entry in samples:
        difficulty = entry.get('difficulty_level')
        if difficulty is not None:
            final_difficulty_counts[difficulty] += 1
        
        result = entry.get('result', {})
        if result.get('solved', False):
            solved_count += 1
        else:
            unsolved_count += 1
    
    for level in sorted(final_difficulty_counts.keys()):
        print(f"  Level {level}: {final_difficulty_counts[level]} samples")
    print(f"  Solved: {solved_count}, Unsolved: {unsolved_count}")


if __name__ == '__main__':
    main()
