#!/usr/bin/env python3
"""
MATH-500 Dataset Tester
Loads and tests the HuggingFaceH4/MATH-500 dataset
"""

from datasets import load_dataset
from datetime import datetime


def load_math500():
    """Load the MATH-500 dataset from Hugging Face"""
    print("=" * 70)
    print("📦 Loading MATH-500 Dataset...")
    print("=" * 70)
    
    ds = load_dataset("HuggingFaceH4/MATH-500")
    return ds


def test_dataset_structure(ds):
    """Test and display the dataset structure"""
    print("\n" + "=" * 70)
    print("🔍 Test 1: Dataset Structure")
    print("=" * 70)
    
    print(f"\nDataset splits: {list(ds.keys())}")
    
    for split in ds.keys():
        print(f"\n  Split: '{split}'")
        print(f"    - Number of examples: {len(ds[split])}")
        print(f"    - Features/Columns: {ds[split].column_names}")
        print(f"    - Feature types: {dict(ds[split].features)}")


def test_sample_examples(ds, num_samples=3):
    """Test and display sample examples from the dataset"""
    print("\n" + "=" * 70)
    print(f"📋 Test 2: Sample Examples (showing {num_samples})")
    print("=" * 70)
    
    split = list(ds.keys())[0]
    data = ds[split]
    
    for i in range(min(num_samples, len(data))):
        example = data[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"  Problem: {example.get('problem', 'N/A')[:200]}...")
        print(f"  Level: {example.get('level', 'N/A')}")
        print(f"  Subject: {example.get('subject', 'N/A')}")
        answer = example.get('answer', 'N/A')
        print(f"  Answer: {answer[:100]}{'...' if len(str(answer)) > 100 else ''}")
        solution = example.get('solution', 'N/A')
        if solution != 'N/A':
            print(f"  Solution: {solution[:150]}...")


def test_dataset_integrity(ds):
    """Test dataset integrity - check for missing values"""
    print("\n" + "=" * 70)
    print("✅ Test 3: Dataset Integrity Check")
    print("=" * 70)
    
    all_passed = True
    
    for split in ds.keys():
        data = ds[split]
        columns = data.column_names
        
        print(f"\n  Split: '{split}' ({len(data)} examples)")
        
        for col in columns:
            none_count = sum(1 for item in data[col] if item is None)
            empty_count = sum(1 for item in data[col] if isinstance(item, str) and item.strip() == "")
            
            status = "✅" if none_count == 0 and empty_count == 0 else "⚠️"
            if none_count > 0 or empty_count > 0:
                all_passed = False
            
            print(f"    {status} Column '{col}': {none_count} nulls, {empty_count} empty strings")
    
    if all_passed:
        print("\n  🎉 All integrity checks passed!")
    else:
        print("\n  ⚠️ Some issues found - check warnings above")


def test_problem_difficulty_distribution(ds):
    """Test and display the difficulty level distribution"""
    print("\n" + "=" * 70)
    print("📊 Test 4: Problem Difficulty Distribution")
    print("=" * 70)
    
    for split in ds.keys():
        data = ds[split]
        
        if 'level' in data.column_names:
            levels = {}
            for item in data['level']:
                levels[item] = levels.get(item, 0) + 1
            
            print(f"\n  Split: '{split}'")
            for level in sorted(levels.keys()):
                bar = "█" * (levels[level] // 5)
                print(f"    Level {level}: {levels[level]:4d} {bar}")


def test_problem_subject_distribution(ds):
    """Test and display the problem subject distribution"""
    print("\n" + "=" * 70)
    print("📊 Test 5: Problem Subject Distribution")
    print("=" * 70)
    
    for split in ds.keys():
        data = ds[split]
        
        if 'subject' in data.column_names:
            subjects = {}
            for item in data['subject']:
                subjects[item] = subjects.get(item, 0) + 1
            
            print(f"\n  Split: '{split}'")
            for subject, count in sorted(subjects.items(), key=lambda x: -x[1]):
                bar = "█" * (count // 2)
                print(f"    {subject:25s}: {count:4d} {bar}")


def test_answer_validation(ds, num_to_validate=10):
    """Test that answers can be extracted and validated"""
    print("\n" + "=" * 70)
    print(f"🧮 Test 6: Answer Extraction Validation ({num_to_validate} samples)")
    print("=" * 70)
    
    split = list(ds.keys())[0]
    data = ds[split]
    
    valid_count = 0
    invalid_count = 0
    
    for i in range(min(num_to_validate, len(data))):
        example = data[i]
        answer = example.get('answer', '')
        solution = example.get('solution', '')
        
        has_answer = answer is not None and str(answer).strip() != ""
        answer_in_solution = str(answer) in solution if solution and has_answer else False
        
        if has_answer:
            valid_count += 1
            status = "✅"
        else:
            invalid_count += 1
            status = "❌"
        
        print(f"  {status} Example {i + 1}: Answer present={has_answer}, In solution={answer_in_solution}")
    
    print(f"\n  Summary: {valid_count} valid, {invalid_count} invalid out of {num_to_validate}")


def run_all_tests():
    """Run all dataset tests"""
    print("🧪 MATH-500 Dataset Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    ds = load_math500()
    
    # Run all tests
    test_dataset_structure(ds)
    test_sample_examples(ds, num_samples=3)
    test_dataset_integrity(ds)
    test_problem_difficulty_distribution(ds)
    test_problem_subject_distribution(ds)
    test_answer_validation(ds, num_to_validate=10)
    
    print("\n" + "=" * 70)
    print("🏁 All tests completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return ds


if __name__ == "__main__":
    run_all_tests()