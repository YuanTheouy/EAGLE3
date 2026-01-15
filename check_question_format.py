"""Check the format of question files."""

import json
import sys
from fastchat.llm_judge.common import load_questions

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_question_format.py <question_file>")
        return
    
    question_file = sys.argv[1]
    questions = load_questions(question_file, 0, 1)
    
    print(f"Loaded {len(questions)} questions")
    if questions:
        question = questions[0]
        print("\nQuestion structure:")
        print(json.dumps(question, indent=2, ensure_ascii=False))
        
        print("\nKeys in question:")
        for key in question.keys():
            print(f"- {key}: {type(question[key])}")

if __name__ == "__main__":
    main()
