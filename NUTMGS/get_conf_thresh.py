import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def get_thresh(dir, min_acc):
    dir = Path(dir)
    file = dir / 'predictions' / 'eval_predictions.csv'
    if not file.exists():
        return 0.0
    
    csv = pd.read_csv(file)
    correct = csv['correct']
    conf = csv['confidence']

    for t in range(500, 700, 1):
        thresh = t / 100000
        acc = np.sum([1 for idx in range(len(correct)) if correct[idx] and conf[idx] >= thresh]) / \
                np.sum([1 for idx in range(len(correct)) if conf[idx] >= thresh])
        if acc >= min_acc:
            return thresh

def main():
    parser = argparse.ArgumentParser(description="Smart merge of Rajasthan dataset tranches")
    parser.add_argument("--dir", required=True,
                       help="The directory for a bio-clip model evaluation")
    
    args = parser.parse_args()
    print(get_thresh(args.dir, 0.85))

if __name__ == "__main__":
    main()
    