#!/usr/bin/env python3
"""
Custom Word Problem Evaluation Script
Evaluates the results from the custom problem solver
"""

import json
import os
from collections import defaultdict
from datetime import datetime


def load_results(filepath):
    """Load results from JSON file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_results(data):
    """Evaluate and analyze the results"""
    
    print("=" * 70)
    print("📊 Custom Word Problem Evaluation Report")
    print("=" * 70)
    
    if 'timestamp' in data:
        print(f"\n🕐 Run Date: {data['timestamp']}")
    if 'model' in data:
        print(f"🤖 Model: {data['model']}")
    
    results = data.get('results', [])
    
    if not results:
        print("❌ No results found!")
        return
    
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    incorrect = sum(1 for r in results if not r.get('correct', False) and 'ERROR' not in str(r.get('model_answer', '')) and r.get('model_answer', '') != 'TIMEOUT')
    errors = sum(1 for r in results if 'ERROR' in str(r.get('model_answer', '')) or r.get('model_answer', '') == 'TIMEOUT')
    
    print(f"\n{'=' * 70}")
    print(f"📈 OVERALL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total Problems: {total}")
    print(f"  ✅ Correct: {correct}")
    print(f"  ❌ Incorrect: {incorrect}")
    print(f"  ⚠️ Errors/Timeouts: {errors}")
    
    if correct + incorrect > 0:
        accuracy = (correct / (correct + incorrect)) * 100
        print(f"\n  🎯 Accuracy (excluding errors): {correct}/{correct + incorrect} = {accuracy:.2f}%")

    # Chapter-wise analysis
    print(f"\n{'=' * 70}")
    print(f"📚 CHAPTER-WISE ANALYSIS")
    print(f"{'=' * 70}")
    
    chapter_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'errors': 0, 'total': 0})
    
    for r in results:
        chapter = r.get('chapter', 'Unknown')
        chapter_stats[chapter]['total'] += 1
        
        if r.get('correct', False):
            chapter_stats[chapter]['correct'] += 1
        elif 'ERROR' in str(r.get('model_answer', '')) or r.get('model_answer', '') == 'TIMEOUT':
            chapter_stats[chapter]['errors'] += 1
        else:
            chapter_stats[chapter]['incorrect'] += 1
    
    print(f"\n  {'Chapter':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 58}")
    
    for chapter in sorted(chapter_stats.keys()):
        stats = chapter_stats[chapter]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {chapter:<30} {stats['correct']:>8} {stats['total']:>8} {acc:>9.1f}%")
    
    # Section-wise analysis
    print(f"\n{'=' * 70}")
    print(f"📈 SECTION-WISE ANALYSIS")
    print(f"{'=' * 70}")
    
    section_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'errors': 0, 'total': 0})
    
    for r in results:
        section = r.get('section', 'Unknown')
        section_stats[section]['total'] += 1
        
        if r.get('correct', False):
            section_stats[section]['correct'] += 1
        elif 'ERROR' in str(r.get('model_answer', '')) or r.get('model_answer', '') == 'TIMEOUT':
            section_stats[section]['errors'] += 1
        else:
            section_stats[section]['incorrect'] += 1
    
    print(f"\n  {'Section':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 58}")
    
    for section in sorted(section_stats.keys()):
        stats = section_stats[section]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {section:<30} {stats['correct']:>8} {stats['total']:>8} {acc:>9.1f}%")
    
    # Error analysis
    print(f"\n{'=' * 70}")
    print(f"⚠️ ERROR ANALYSIS")
    print(f"{'=' * 70}")
    
    error_types = defaultdict(int)
    for r in results:
        answer = r.get('model_answer', '')
        if 'ERROR' in str(answer):
            error_types['Model Error'] += 1
        elif answer == 'TIMEOUT':
            error_types['Timeout'] += 1
    
    if error_types:
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
    else:
        print("  No errors!")
    
    # Sample incorrect answers
    print(f"\n{'=' * 70}")
    print(f"❌ SAMPLE INCORRECT ANSWERS (first 10)")
    print(f"{'=' * 70}")
    
    incorrect_results = [r for r in results if not r.get('correct', False) and 'ERROR' not in str(r.get('model_answer', '')) and r.get('model_answer', '') != 'TIMEOUT']
    
    for i, r in enumerate(incorrect_results[:10]):
        print(f"\n  [{i + 1}] Problem: {r['question'][:80]}...")
        print(f"      Expected: {r['ground_truth']}")
        print(f"      Got:      {r['model_answer'][:100]}")
    
    # Sample correct answers
    print(f"\n{'=' * 70}")
    print(f"✅ SAMPLE CORRECT ANSWERS (first 10)")
    print(f"{'=' * 70}")
    
    correct_results = [r for r in results if r.get('correct', False)]
    
    for i, r in enumerate(correct_results[:10]):
        print(f"\n  [{i + 1}] Problem: {r['question'][:80]}...")
        print(f"      Answer: {r['ground_truth']}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_problems': total,
        'correct': correct,
        'incorrect': incorrect,
        'errors': errors,
        'accuracy': (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0,
        'overall_accuracy': (correct / total * 100) if total > 0 else 0,
        'chapter_stats': {k: dict(v) for k, v in chapter_stats.items()},
        'section_stats': {k: dict(v) for k, v in section_stats.items()},
    }
    
    summary_file = 'custom_evaluation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"💾 Summary saved to: {summary_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate custom word problem solver results")
    parser.add_argument('--file', '-f', type=str, default='custom_results.json', help='Results file to evaluate')
    
    args = parser.parse_args()
    
    data = load_results(args.file)
    if data:
        evaluate_results(data)
        accuracy = (correct / (correct + incorrect)) * 100
        print(f"\n  🎯 Accuracy (excluding errors): {correct}/{correct + incorrect} = {accuracy:.2f}%")
    
    if total > 0:
        overall_accuracy = (correct / total) * 100
        print(f"  🎯 Overall Accuracy: {correct}/{total} = {overall_accuracy:.2f}%")