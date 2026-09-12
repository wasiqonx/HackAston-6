<div align="center">

# 🧮 Spark-X2.5-1.7B Math Problem Solver

### Comprehensive Math Problem Solving & Evaluation System

![Model](https://img.shields.io/badge/Model-Spark--X2.5--1.7B-blueviolet)
![Dataset](https://img.shields.io/badge/Datasets-MATH--500%20%7C%20Custom%20Problems-orange)
![Language](https://img.shields.io/badge/Language-Python-yellowgreen)
![Accuracy](https://img.shields.io/badge/MATH--500%20Accuracy-73.24%25-success)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Model Information](#-model-information)
- [Dataset Format](#-dataset-format)
- [Evaluation Metrics](#-evaluation-metrics)
- [Test Results](#-test-results)
- [Conclusion](#-conclusion)

---

## 📖 Overview

A comprehensive math problem solving and evaluation system using the **Spark-X2.5-1.7B** language model. The system evaluates model performance on standardized benchmarks and custom word problems, providing detailed accuracy metrics and error analysis.

---

## 📁 Project Structure

```
HackAston-6/
├── .venv/                          # Python virtual environment
├── MATH-500/                       # MATH-500 benchmark evaluation
│   ├── evaluate.py                 # MATH-500 evaluation script
│   ├── solve_math500.py            # MATH-500 solver
│   ├── results_full.json           # Full evaluation results
│   ├── evaluation_summary.json     # Evaluation summary
│   └── solve_log.txt               # Solver log
├── Custom/                         # Custom test directory
├── custom_word_problem.csv         # Custom word problems (CSV format)
├── custom_word_problem.jsonl       # Custom word problems (JSONL format)
├── solve_custom_problems.py        # Solver for custom word problems
├── evaluate_custom.py              # Evaluator for custom problems
├── custom_results.json             # Custom problems results
└── README.md                       # This file
```

---

## 🚀 Quick Start

```bash
cd /Users/wasiq/Desktop/WW/ProjectLOL/HackAston-6/
source .venv/bin/activate
pip install datasets
```

### Solve Custom Word Problems

```bash
# Solve from JSONL file (default)
.venv/bin/python solve_custom_problems.py --file custom_word_problem.jsonl --output custom_results.json

# Solve from CSV file
.venv/bin/python solve_custom_problems.py --file custom_word_problem.csv --output custom_results.json
```

### Evaluate Results

```bash
.venv/bin/python evaluate_custom.py --file custom_results.json
```

---

## ✨ Features

### 1. Custom Word Problem Solver
Solves 100 custom word problems covering:
- **Average**
- **Percentage**
- **Ratio and Proportion**

### 2. Comprehensive Scoring System
| Strategy | Description |
|----------|-------------|
| **Direct Matching** | Exact answer comparison |
| **Normalization** | Removes formatting differences |
| **Fraction Evaluation** | Evaluates `\frac{a}{b}` expressions |
| **Numeric Comparison** | Compares decimal values with tolerance |
| **Boxed Format Handling** | Extracts answers from `\boxed{}` format |

### 3. Detailed Evaluation Reports
- Overall accuracy metrics
- Chapter-wise performance analysis
- Section-wise performance analysis
- Error analysis (timeouts, model errors)
- Sample correct/incorrect answers

---

## 🤖 Model Information

| Component | Details |
|-----------|---------|
| **Model** | Spark-X2.5-1.7B |
| **Device** | GPU |
| **Dtype** | bfloat16 |
| **Max Tokens** | 30,000 |
| **Temperature** | 0.7 |
| **Timeout** | 300 seconds per problem |

---

## 📄 Dataset Format

### Custom Word Problems (JSONL)
```json
{
  "id": 1,
  "chapter": "Average",
  "section": "Mains",
  "question": "The average cost of 5 books...",
  "answer": "₹900",
  "verification": "verified"
}
```

---

## 📏 Evaluation Metrics

### Comparison Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Direct Match** | Compares normalized answers directly |
| 2 | **LaTeX Fraction Evaluation** | Extracts and evaluates `\frac{numerator}{denominator}` |
| 3 | **Mixed Number Handling** | Handles `33\frac{1}{3}` format |
| 4 | **Numeric Extraction** | Compares decimal values with tolerance (±0.01) |
| 5 | **Normalization** | Removes formatting (currency, units, whitespace, trailing periods) |

### Answer Comparison Examples

| Model Answer | Expected | Result | Reason |
|--------------|----------|--------|--------|
| `63.6%.` | `63.6%` | ✅ | Trailing period removed |
| `\boxed{176 \text{ sq cm}` | `176 sq cm` | ✅ | Boxed format handled |
| `\frac{846}{12}` | `70.5` | ✅ | Fraction evaluated (846÷12=70.5) |
| `\frac{100}{3}` | `33.33` | ✅ | Fraction evaluated |
| `35 years.` | `35 years` | ✅ | Trailing period removed |
| `33\frac{1}{3}` | `33.33` | ✅ | Mixed number handled |

### Metrics Overview

| Metric | Description |
|--------|-------------|
| **Total Problems** | Number of problems attempted |
| **Correct** | Correctly answered problems |
| **Incorrect** | Incorrectly answered problems |
| **Errors/Timeouts** | Timeouts or model errors |
| **Accuracy** | Percentage of correct answers (excluding errors) |

---

## 📊 Test Results

### MATH-500 Dataset (500 Problems)

<details open>
<summary><b>Overall Performance</b></summary>

| Metric | Value |
|--------|-------|
| **Correct** | 249 |
| **Incorrect** | 91 |
| **Errors/Timeouts** | 160 |
| **Accuracy (excluding errors)** | **73.24%** |
| **Overall Accuracy** | **49.80%** |

</details>

<details>
<summary><b>Subject-wise Performance</b></summary>

| Subject | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Number Theory | 46 | 62 | 74.2% |
| Algebra | 84 | 124 | 67.7% |
| Prealgebra | 39 | 82 | 47.6% |
| Intermediate Algebra | 34 | 97 | 35.1% |
| Counting & Probability | 14 | 38 | 36.8% |
| Geometry | 14 | 41 | 34.1% |
| Precalculus | 18 | 56 | 32.1% |

</details>

<details>
<summary><b>Difficulty Level Performance</b></summary>

| Level | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Level 1 | 35 | 43 | 81.4% |
| Level 2 | 59 | 90 | 65.6% |
| Level 3 | 57 | 105 | 54.3% |
| Level 4 | 61 | 128 | 47.7% |
| Level 5 | 37 | 134 | 27.6% |

</details>

---

### Custom Word Problems (100 Problems)

<details open>
<summary><b>Overall Performance</b></summary>

| Metric | Value |
|--------|-------|
| **Correct** | 28 |
| **Incorrect** | 50 |
| **Errors/Timeouts** | 22 |
| **Accuracy (excluding errors)** | **35.90%** |
| **Overall Accuracy** | **28.00%** |

</details>

---

## 📈 Comparison Summary

| Dataset | Problems | Correct | Accuracy (excl. errors) | Overall Accuracy |
|---------|----------|---------|-------------------------|------------------|
| MATH-500 | 500 | 249 | 73.24% | 49.80% |
| Custom Word Problems | 100 | 28 | 35.90% | 28.00% |

---

## 🎯 Key Observations

1. **MATH-500 Performance:** The model performs well on easier problems (Level 1: 81.4%) but struggles with harder ones (Level 5: 27.6%)

2. **Custom Problems:** Lower accuracy suggests the model struggles with word problems requiring multi-step reasoning

3. **Timeout Issues:** Both datasets have significant timeout issues (32% for MATH-500, 22% for custom)

4. **Strongest Subject:** Number Theory (74.2%) for MATH-500

---

## 📝 Conclusion

The **Spark-X2.5-1.7B** model demonstrates the following characteristics:

### ✅ Strengths
- **Strong on Standardized Tests:** 73.24% accuracy on MATH-500 (excluding errors)
- **Easiest Problems Solved Well:** 81.4% accuracy on Level 1 difficulty problems
- **Best at Number Theory:** 74.2% accuracy in this subject area

### ❌ Weaknesses
- **Struggles with Hard Problems:** Only 27.6% accuracy on Level 5 difficulty
- **Word Problem Challenges:** Only 35.90% accuracy on custom word problems
- **Timeout Issues:** 32% of MATH-500 problems hit timeout limits
- **Inconsistent Across Subjects:** Ranges from 74.2% (Number Theory) to 32.1% (Precalculus)

### 📊 Overall Assessment
The model shows promise for standardized math problems but requires improvement on complex multi-step reasoning tasks and word problems. Reducing timeout issues and improving performance on higher difficulty levels are key areas for future work.

---

<div align="center">

---

**Created:** September 2026 | **Model:** Spark-X2.5-1.7B | **Datasets:** MATH-500, Custom Word Problems

</div>