import json
import argparse
import re
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

def extract_move(response: str) -> Optional[str]:
    """Extract SAN move from model response."""
    move_match = re.search(r"Move:\s*([A-Za-z0-9+#=-]+)", response)
    if move_match:
        return move_match.group(1)
    return None

def extract_confidence(response: str) -> float:
    """Extract confidence from model response."""
    confidence_match = re.search(r"Confidence:\s*(\d+)", response)
    if confidence_match:
        return float(confidence_match.group(1))
    return 100.0  # Default confidence

def judge_move(predicted: str, correct: str) -> bool:
    """Compare predicted move with correct move."""
    if not predicted:
        return False
    return predicted.strip().rstrip('+#') == correct.strip()

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy and confidence interval."""
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate 95% confidence interval
    z = 1.96  # 95% confidence
    interval = z * np.sqrt((accuracy * (1 - accuracy)) / total_predictions)
    
    # Calculate calibration error if confidences are available
    confidences = [r['confidence'] / 100.0 for r in results]
    calibration_error = None
    if confidences:
        calibration_error = abs(np.mean(confidences) - accuracy)
    
    return {
        'accuracy': accuracy,
        'confidence_interval': interval,
        'calibration_error': calibration_error,
        'total_samples': total_predictions
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()
    
    # Load dataset and predictions
    with open(args.dataset) as f:
        dataset = {pos['id']: pos for pos in json.load(f)}
    
    with open(args.predictions) as f:
        predictions = json.load(f)
    
    # Set output path
    output_filepath = Path(args.predictions).name.replace('predictions_', 'judged_predictions_')
    output_path = Path(__file__).parent.parent / "outputs" / "judged_results" / output_filepath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each prediction
    results = []
    for pos_id, pred in predictions.items():
        if pos_id not in dataset:
            print(f"Position {pos_id} not found in dataset, skipping")
            continue
            
        position = dataset[pos_id]
        model_response = pred['response']
        
        extracted_move = extract_move(model_response)
        confidence = extract_confidence(model_response)
        correct = judge_move(extracted_move, position['answer'])
        
        judged_result = {
            'model_answer': extracted_move,
            'correct_answer': position['answer'],
            'correct': correct,
            'confidence': confidence
        }
        
        # Add to results list and update predictions
        results.append(judged_result)
        pred['judge_response'] = judged_result
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save judged predictions
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Print metrics
    model_name = Path(args.predictions).stem.replace('predictions_', '')
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']*100:.1f}% ± {metrics['confidence_interval']*100:.1f}%")
    print(f"Total samples: {metrics['total_samples']}")
    if metrics['calibration_error'] is not None:
        print(f"Calibration error: {metrics['calibration_error']*100:.1f}%")

if __name__ == "__main__":
    main()