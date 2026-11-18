"""Compare raw numeric prediction inputs against a training-only reference."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=ROOT / 'logs/predictions.jsonl')
    parser.add_argument('--reference', type=Path, default=ROOT / 'model/reference.json')
    parser.add_argument('--min-samples', type=int, default=30)
    args = parser.parse_args()
    if args.min_samples < 2:
        parser.error('min-samples must be at least 2')
    reference = json.loads(args.reference.read_text())
    paths = sorted(args.log.parent.glob(args.log.name + '.*')) + ([args.log] if args.log.exists() else [])
    rows = [json.loads(line)['inputs'] for path in paths for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < args.min_samples:
        print(json.dumps({'status': 'insufficient_data', 'samples': len(rows), 'minimum': args.min_samples}, indent=2))
        return
    summary = {}
    for col, stats in reference.items():
        values = np.array([row[col] for row in rows], dtype=float)
        mean, std = float(values.mean()), float(values.std())
        shift = abs(mean - stats['mean']) / max(stats['std'], 1e-9)
        ratio = std / max(stats['std'], 1e-9)
        summary[col] = {'mean': mean, 'std': std, 'mean_shift_in_training_std': shift,
                        'std_ratio': ratio, 'flag': bool(shift > .5 or ratio < .5 or ratio > 2)}
    print(json.dumps({'status': 'review' if any(v['flag'] for v in summary.values()) else 'ok',
                      'samples': len(rows), 'features': summary}, indent=2))


if __name__ == '__main__':
    main()
