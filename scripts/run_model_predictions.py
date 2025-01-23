import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import anthropic
from tqdm.asyncio import tqdm_asyncio

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a chess expert. Analyze the given chess position and provide the best next move in Standard Algebraic Notation (SAN).
Your response must be in this format:
Explanation: {your brief analysis}
Move: {best move in SAN}
Confidence: {0-100}"""

def query_model(position: Dict, args: argparse.Namespace) -> Dict:
    try:
        response = client.messages.create(
            model=args.model,
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Given this chess position: {position['question']}\nWhat is the best next move?"}]
        )
        return {
            "model": args.model,
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    except Exception as e:
        print(f"Error processing position {position['id']}: {str(e)}")
        return None

def process_positions(positions: List[Dict], args: argparse.Namespace) -> Dict:
    results = {}
    for position in tqdm_asyncio(positions):
        result = query_model(position, args)
        if result:
            results[position['id']] = result
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229")
    parser.add_argument("--max_completion_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=5)
    args = parser.parse_args()
    
    output_filepath = f"predictions_{args.model}.json"
    output_path = Path(__file__).parent.parent / "outputs" / "model_predictions" / output_filepath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_predictions = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing_predictions = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not read existing predictions file ({str(e)}). Starting fresh.")
            existing_predictions = {}
    
    with open(args.dataset) as f:
        positions = json.load(f)
    
    positions_to_process = [
        pos for pos in positions 
        if pos['id'] not in existing_predictions
    ]
    
    if not positions_to_process:
        print("No new positions to process.")
        return
        
    print(f"Processing {len(positions_to_process)} positions...")
    
    results = process_positions(positions_to_process, args)
    existing_predictions.update(results)
    
    with open(output_path, 'w') as f:
        json.dump(existing_predictions, f, indent=2)
    
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()