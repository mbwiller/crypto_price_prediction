import pandas as pd

train = pd.read_parquet(r'C:\Users\Matt Willer\Downloads\drw-crypto-market-prediction\train.parquet', engine='pyarrow')
test = pd.read_parquet(r'C:\Users\Matt Willer\Downloads\drw-crypto-market-prediction\test.parquet', engine='pyarrow')

print("=== Train data head ===")
print(train.head())

print("\n=== Test data head ===")
print(test.head())

train.head(1000).to_csv('train_first_1000.csv', index=True)

# same for test
test.head(1000).to_csv('test_first_1000.csv', index=True)