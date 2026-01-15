#!/usr/bin/env python3
"""Evaluate VQA answers using standard VQA metrics."""

import json
from collections import Counter
import argparse

def calculate_vqa_accuracy(model_answer, reference_answers):
    """
    Calculate VQA accuracy for a single question.
    VQA accuracy is defined as the proportion of reference answers that match the model answer.
    If the model answer matches at least 30% of the reference answers, it's considered correct.
    """
    if not reference_answers:
        return 0.0
    
    model_answer_lower = model_answer.lower().strip()
    reference_lower = [ans.lower().strip() for ans in reference_answers]
    
    # Count occurrences of each reference answer
    answer_counts = Counter(reference_lower)
    
    # Calculate accuracy
    if model_answer_lower in answer_counts:
        return min(1.0, answer_counts[model_answer_lower] / 3.0)  # Max 1.0, need at least 3 votes for full credit
    else:
        return 0.0

def evaluate_vqa(answers_file, questions_file):
    """
    Evaluate VQA answers against reference answers.
    """
    # Load reference questions
    print("Loading reference questions...")
    references = {}
    with open(questions_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            references[data["question_id"]] = data["reference"]
    
    print(f"Loaded {len(references)} reference questions")
    
    # Load model answers
    print("Loading model answers...")
    model_answers = {}
    with open(answers_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            qid = data["question_id"]
            # Get the first choice (assuming num_choices=1)
            model_answer = data["choices"][0]["turns"][0]
            model_answers[qid] = model_answer
    
    print(f"Loaded {len(model_answers)} model answers")
    
    # Calculate metrics
    print("Calculating metrics...")
    total_accuracy = 0.0
    exact_match = 0
    total = 0
    
    for qid, ref_answers in references.items():
        if qid not in model_answers:
            continue
        
        model_answer = model_answers[qid]
        accuracy = calculate_vqa_accuracy(model_answer, ref_answers)
        total_accuracy += accuracy
        
        # Check exact match
        if model_answer.lower().strip() in [ans.lower().strip() for ans in ref_answers]:
            exact_match += 1
        
        total += 1
    
    # Compute final metrics
    avg_accuracy = total_accuracy / total if total > 0 else 0
    exact_match_ratio = exact_match / total if total > 0 else 0
    
    # Print results
    print("\n=== VQA Evaluation Results ===")
    print(f"Total samples evaluated: {total}")
    print(f"Average VQA accuracy: {avg_accuracy:.4f}")
    print(f"Exact match ratio: {exact_match_ratio:.4f}")
    print(f"Number of exact matches: {exact_match}")
    
    return {
        "total_samples": total,
        "avg_accuracy": avg_accuracy,
        "exact_match_ratio": exact_match_ratio,
        "exact_matches": exact_match
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA answers")
    parser.add_argument("--answers-file", required=True, help="Path to model answers JSONL file")
    parser.add_argument("--questions-file", required=True, help="Path to questions with references JSONL file")
    
    args = parser.parse_args()
    
    evaluate_vqa(args.answers_file, args.questions_file)

if __name__ == "__main__":
    main()
