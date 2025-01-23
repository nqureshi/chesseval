# Chess Evaluation Suite

A tool to evaluate AI model performance on chess position analysis.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-key'
```

## Usage

Generate predictions:
```bash
cd scripts
python run_model_predictions.py --dataset ../data/chess_positions.json --model claude-3-opus-20240229
```

Evaluate predictions:
```bash
python run_judge_results.py --dataset ../data/chess_positions.json --predictions ../outputs/model_predictions/predictions_claude-3-opus-20240229.json
```

## Data Format

Chess positions in `data/chess_positions.json`:
```json
[{
  "id": "pos1",
  "question": "FEN: [fen_string]",
  "answer": "[move in SAN]",
  "answer_type": "exact_match",
  "image": ""
}]
```

## Output

Predictions stored in `outputs/model_predictions/`
Evaluation results in `outputs/judged_results/`