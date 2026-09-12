#!/usr/bin/env python3
"""Custom Word Problem Solver using Spark-X2.5-1.7B"""

import subprocess
import sys
import os
import json
import csv
import re
from datetime import datetime

SPARK_MLX_PATH = "/Users/wasiq/Projects/Private/Spark/Spark-MLX-LLM/.venv/bin/spark-mlx-generate"
MODEL_NAME = "XHToken/Spark-X2.5-1.7B"
MAX_TOKENS = 30000


def load_problems_from_csv(filepath):
    problems = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append(row)
    return problems


def load_problems_from_jsonl(filepath):
    problems = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def solve_with_spark(problem, max_tokens=MAX_TOKENS):
    prompt = f"Solve the following word problem step by step. Show your work clearly and provide the final answer.\n\nProblem: {problem}\n\nSolution:"
    cmd = [
        SPARK_MLX_PATH, "--model", MODEL_NAME, "--prompt", prompt,
        "--max-tokens", str(max_tokens), "--temp", "0.7",
        "--device", "gpu", "--dtype", "bfloat16"
    ]
    try:
        result = subprocess.run(cmd, input="y\n", capture_output=True, text=True, check=True, timeout=300)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_answer_from_response(response):
    if "Do you wish to run the custom code?" in response:
        match = re.search(r'\[y/N\]\s*=+\s*(.+)', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
    patterns = [
        r'\\boxed\{([^}]+)\}', r'\\boxed\{(.+?)\}',
        r'\\dfrac\{[^}]+\}\{[^}]+\}', r'\\frac\{[^}]+\}\{[^}]+\}',
        r'[Tt]he answer is[:\s]+(.+?)$', r'[Tt]he final answer is[:\s]+(.+?)$',
        r'[Aa]nswer:\s*(.+?)$', r'Final Answer:\s*(.+?)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            answer = match.group(0) if 'frac' in pattern or 'boxed' in pattern else match.group(1)
            return answer.strip()
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('Thus'):
            return line
    return response.strip()


def normalize_answer(answer):
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove LaTeX formatting
    answer = answer.replace('\\', '').replace('{', '').replace('}', '')
    answer = answer.replace('left', '').replace('right', '')
    # Remove currency and units
    answer = answer.replace('₹', '').replace('rs.', '').replace('rs', '')
    answer = answer.replace('approximately', '').replace('approx', '')
    # Remove boxed format
    answer = answer.replace('boxed', '')
    # Remove text format
    answer = answer.replace('text', '')
    # Remove trailing period
    answer = answer.rstrip('.')
    # Remove whitespace
    answer = re.sub(r'\s+', '', answer)
    return answer


def check_answer(model_answer, ground_truth):
    norm_model = normalize_answer(model_answer)
    norm_truth = normalize_answer(ground_truth)
    
    # Direct match
    if norm_model == norm_truth:
        return True
    
    # Check if ground truth is contained in model answer
    if norm_truth in norm_model:
        return True
    
    # Check if model answer is contained in ground truth
    if norm_model in norm_truth:
        return True
    
    # Handle fraction evaluation - extract from original LaTeX format
    try:
        # Match \frac{num}{den} pattern in original answer
        latex_frac = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', model_answer)
        if latex_frac:
            num = int(latex_frac.group(1))
            den = int(latex_frac.group(2))
            if den != 0:
                decimal_val = num / den
                # Check against expected answer
                truth_nums = re.findall(r'[\d.]+', ground_truth)
                for truth_num in truth_nums:
                    try:
                        if abs(decimal_val - float(truth_num)) < 0.01:
                            return True
                    except ValueError:
                        pass
                # Also check rounded values
                if truth_nums and abs(round(decimal_val, 2) - float(truth_nums[0])) < 0.01:
                    return True
        
        # Handle mixed numbers in LaTeX: \boxed{33\frac{1}{3}} or similar
        latex_mixed = re.search(r'(\d+)\\frac\{(\d+)\}\{(\d+)\}', model_answer)
        if latex_mixed:
            whole = int(latex_mixed.group(1))
            num = int(latex_mixed.group(2))
            den = int(latex_mixed.group(3))
            if den != 0:
                decimal_val = whole + num / den
                truth_nums = re.findall(r'[\d.]+', ground_truth)
                for truth_num in truth_nums:
                    try:
                        if abs(decimal_val - float(truth_num)) < 0.01:
                            return True
                    except ValueError:
                        pass
    except (ValueError, ZeroDivisionError):
        pass
    
    # Handle simple fraction format like "846/12"
    simple_frac = re.search(r'(\d+)/(\d+)', model_answer)
    if simple_frac:
        num = int(simple_frac.group(1))
        den = int(simple_frac.group(2))
        if den != 0:
            decimal_val = num / den
            truth_nums = re.findall(r'[\d.]+', ground_truth)
            for truth_num in truth_nums:
                try:
                    if abs(decimal_val - float(truth_num)) < 0.01:
                        return True
                except ValueError:
                    pass
    
    # Extract numeric values for comparison
    model_nums = re.findall(r'[\d,]+\.?\d*', model_answer)
    truth_nums = re.findall(r'[\d,]+\.?\d*', ground_truth)
    
    if model_nums and truth_nums:
        model_last = model_nums[-1].replace(',', '')
        truth_last = truth_nums[-1].replace(',', '')
        try:
            if abs(float(model_last) - float(truth_last)) < 0.01:
                return True
        except ValueError:
            pass
    
    return False


def run_solver(problems, output_file="custom_results.json"):
    print("=" * 70)
    print("📝 Custom Word Problem Solver using Spark-X2.5-1.7B")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    total_problems = len(problems)
    print(f"📊 Total problems: {total_problems}")
    print("-" * 70)
    results = []
    correct = 0
    incorrect = 0
    errors = 0
    for i, problem in enumerate(problems):
        problem_text = problem.get('question', '')
        ground_truth = problem.get('answer', '')
        problem_id = problem.get('id', i + 1)
        chapter = problem.get('chapter', 'Unknown')
        section = problem.get('section', 'Unknown')
        print(f"\n[{i + 1}/{total_problems}] Problem {problem_id} ({chapter} - {section}):")
        print(f"  Q: {problem_text[:100]}...")
        response = solve_with_spark(problem_text)
        if response.startswith("ERROR") or response == "TIMEOUT":
            print(f"  ❌ {response}")
            errors += 1
            is_correct = False
            model_answer = response
        else:
            model_answer = extract_answer_from_response(response)
            is_correct = check_answer(model_answer, ground_truth)
            if is_correct:
                correct += 1
                print(f"  ✅ Correct! Answer: {ground_truth}")
            else:
                incorrect += 1
                print(f"  ❌ Incorrect")
                print(f"     Model answer: {model_answer}")
                print(f"     Expected: {ground_truth}")
        result = {
            "index": i, "id": problem_id, "chapter": chapter, "section": section,
            "question": problem_text, "ground_truth": ground_truth,
            "model_answer": model_answer, "full_response": response, "correct": is_correct
        }
        results.append(result)
        solved = correct + incorrect
        if solved > 0:
            accuracy = (correct / solved) * 100
            print(f"  📈 Running accuracy: {correct}/{solved} = {accuracy:.1f}%")
    print("\n" + "=" * 70)
    print("🏁 SOLVING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"  Total problems attempted: {total_problems}")
    print(f"  ✅ Correct: {correct}")
    print(f"  ❌ Incorrect: {incorrect}")
    print(f"  ⚠️ Errors: {errors}")
    solved = correct + incorrect
    if solved > 0:
        accuracy = (correct / solved) * 100
        print(f"\n  🎯 Accuracy: {correct}/{solved} = {accuracy:.2f}%")
    print(f"\n📚 Chapter-wise Results:")
    chapter_stats = {}
    for r in results:
        chap = r['chapter']
        if chap not in chapter_stats:
            chapter_stats[chap] = {'correct': 0, 'total': 0}
        chapter_stats[chap]['total'] += 1
        if r['correct']:
            chapter_stats[chap]['correct'] += 1
    for chap, stats in sorted(chapter_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {chap:25s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    print(f"\n📈 Section-wise Results:")
    section_stats = {}
    for r in results:
        sec = r['section']
        if sec not in section_stats:
            section_stats[sec] = {'correct': 0, 'total': 0}
        section_stats[sec]['total'] += 1
        if r['correct']:
            section_stats[sec]['correct'] += 1
    for sec, stats in sorted(section_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {sec:25s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    output_data = {
        "timestamp": datetime.now().isoformat(), "model": MODEL_NAME,
        "total_problems": total_problems, "correct": correct, "incorrect": incorrect,
        "errors": errors, "accuracy": (correct / solved * 100) if solved > 0 else 0, "results": results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Solve custom word problems with Spark-X2.5-1.7B")
    parser.add_argument('--file', '-f', type=str, default='custom_word_problem.jsonl', help='Input file (CSV or JSONL)')
    parser.add_argument('--output', '-o', type=str, default='custom_results.json', help='Output file')
    args = parser.parse_args()
    filepath = args.file
    if filepath.endswith('.csv'):
        problems = load_problems_from_csv(filepath)
    elif filepath.endswith('.jsonl'):
        problems = load_problems_from_jsonl(filepath)
    else:
        print("❌ Unsupported file format. Use .csv or .jsonl")
        sys.exit(1)
    print(f"📂 Loaded {len(problems)} problems from {filepath}")
    run_solver(problems, output_file=args.output)