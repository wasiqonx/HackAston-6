#!/usr/bin/env python3
"""
MATH-500 Solver using Spark-X2.5-1.7B
Solves math problems from the MATH-500 dataset using the Spark model
"""

import subprocess
import sys
import os
import json
import re
from datetime import datetime
from datasets import load_dataset

# Configuration - Spark Model
SPARK_MLX_PATH = "/Users/wasiq/Projects/Private/Spark/Spark-MLX-LLM/.venv/bin/spark-mlx-generate"
MODEL_NAME = "XHToken/Spark-X2.5-1.7B"
MAX_TOKENS = 30000


def solve_with_spark(problem, max_tokens=MAX_TOKENS):
    """Send a math problem to Spark-X2.5-1.7B and get the solution"""
    
    prompt = f"""Solve the following math problem step by step. Show your work clearly and provide the final answer.

Problem: {problem}

Solution:"""
    
    cmd = [
        SPARK_MLX_PATH,
        "--model", MODEL_NAME,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.7",
        "--device", "gpu",
        "--dtype", "bfloat16"
    ]
    
    try:
        # Pipe "y" to automatically accept trust_remote_code prompt
        result = subprocess.run(
            cmd, 
            input="y\n",
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=300
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_answer_from_response(response):
    """Extract the final answer from the model's response"""
    # Clean up the response - remove the trust_remote_code prompt if present
    if "Do you wish to run the custom code?" in response:
        # Find the actual response after the prompt
        match = re.search(r'\[y/N\]\s*=+\s*(.+)', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
    
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\{(.+?)\}',
        r'\\dfrac\{[^}]+\}\{[^}]+\}',
        r'\\frac\{[^}]+\}\{[^}]+\}',
        r'[Tt]he answer is[:\s]+(.+?)$',
        r'[Tt]he final answer is[:\s]+(.+?)$',
        r'[Aa]nswer:\s*(.+?)$',
        r'Final Answer:\s*(.+?)$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            answer = match.group(0) if 'frac' in pattern or 'boxed' in pattern else match.group(1)
            return answer.strip()
    
    # Look for LaTeX fractions or expressions at the end
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('Thus'):
            return line
    
    return response.strip()


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = answer.replace('\\', '').replace('{', '').replace('}', '')
    answer = answer.replace('left', '').replace('right', '')
    answer = re.sub(r'\s+', '', answer)
    return answer


def check_answer(model_answer, ground_truth):
    """Check if the model's answer matches the ground truth"""
    norm_model = normalize_answer(model_answer)
    norm_truth = normalize_answer(ground_truth)
    
    if norm_model == norm_truth:
        return True
    if norm_truth in norm_model:
        return True
    if norm_model in norm_truth:
        return True
    
    return False


def run_solver(num_problems=None, start_from=0, output_file="results.json"):
    """Run the solver on MATH-500 dataset"""
    
    print("=" * 70)
    print("🧮 MATH-500 Solver using Spark-X2.5-1.7B")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n📦 Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500")
    data = ds['test']
    
    total_problems = len(data)
    if num_problems:
        end_index = min(start_from + num_problems, total_problems)
    else:
        end_index = total_problems
    
    print(f"📊 Total problems: {total_problems}")
    print(f"🎯 Solving problems {start_from + 1} to {end_index}")
    print("-" * 70)
    
    results = []
    correct = 0
    incorrect = 0
    errors = 0
    
    for i in range(start_from, end_index):
        problem = data[i]
        problem_text = problem['problem']
        ground_truth = problem['answer']
        subject = problem['subject']
        level = problem['level']
        unique_id = problem['unique_id']
        
        print(f"\n[{i + 1}/{end_index}] Problem (Level {level}, {subject}):")
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
            "index": i,
            "unique_id": unique_id,
            "problem": problem_text,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "full_response": response,
            "subject": subject,
            "level": level,
            "correct": is_correct
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
    print(f"  Total problems attempted: {end_index - start_from}")
    print(f"  ✅ Correct: {correct}")
    print(f"  ❌ Incorrect: {incorrect}")
    print(f"  ⚠️ Errors: {errors}")
    
    solved = correct + incorrect
    if solved > 0:
        accuracy = (correct / solved) * 100
        print(f"\n  🎯 Accuracy: {correct}/{solved} = {accuracy:.2f}%")
    
    print(f"\n📚 Subject-wise Results:")
    subject_stats = {}
    for r in results:
        subj = r['subject']
        if subj not in subject_stats:
            subject_stats[subj] = {'correct': 0, 'total': 0}
        subject_stats[subj]['total'] += 1
        if r['correct']:
            subject_stats[subj]['correct'] += 1
    
    for subj, stats in sorted(subject_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {subj:25s}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    
    print(f"\n📈 Level-wise Results:")
    level_stats = {}
    for r in results:
        lvl = r['level']
        if lvl not in level_stats:
            level_stats[lvl] = {'correct': 0, 'total': 0}
        level_stats[lvl]['total'] += 1
        if r['correct']:
            level_stats[lvl]['correct'] += 1
    
    for lvl, stats in sorted(level_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  Level {lvl}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "total_problems": end_index - start_from,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": (correct / solved * 100) if solved > 0 else 0,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve MATH-500 problems with Spark-X2.5-1.7B")
    parser.add_argument('--num', '-n', type=int, help='Number of problems to solve')
    parser.add_argument('--start', '-s', type=int, default=0, help='Start from problem index')
    parser.add_argument('--output', '-o', type=str, default='results.json', help='Output file')
    
    args = parser.parse_args()
    
    run_solver(num_problems=args.num, start_from=args.start, output_file=args.output)