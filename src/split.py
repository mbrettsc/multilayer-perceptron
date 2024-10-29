import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) != 2:
        print('Usage: python split.py <input_file>')
        return
    
    try:
        data = pd.read_csv(sys.argv[1])
    except Exception as e:
        print('Error reading file:', e)
        return

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train.to_csv('../data/processed/train.csv', index=False)
    test.to_csv('../data/processed/test.csv', index=False)


if __name__ == '__main__':
    main()