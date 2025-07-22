import pandas as pd
import numpy as np

def prepare_data():
    """Prepare train and test data for the RL system"""
    
    print("Loading CSV files...")
    train_df = pd.read_csv('train_first_1000.csv')
    test_df = pd.read_csv('test_first_1000.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Since we don't have a 'close' column, we need to create one
    # We'll use the cumulative sum of labels to simulate price movements
    print("\nCreating synthetic close prices...")
    
    # For training data
    if 'close' not in train_df.columns:
        # Start with a base price (e.g., 100)
        base_price = 100.0
        # Use cumulative sum of returns (labels) to create prices
        train_df['close'] = base_price * (1 + train_df['label']).cumprod()
        # Fill any NaN values
        train_df['close'].fillna(method='ffill', inplace=True)
    
    # For test data
    if 'close' not in test_df.columns:
        # Continue from the last training price
        last_train_price = train_df['close'].iloc[-1] if 'close' in train_df.columns else 100.0
        test_df['close'] = last_train_price * (1 + test_df['label']).cumprod()
        test_df['close'].fillna(method='ffill', inplace=True)
    
    # Add high/low columns (required for some technical indicators)
    for df in [train_df, test_df]:
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.001  # 0.1% above close
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.999   # 0.1% below close
    
    # Ensure all required columns have proper data types
    float_columns = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'close', 'high', 'low', 'label']
    float_columns.extend([f'X{i}' for i in range(1, 781)])
    
    for col in float_columns:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0)
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0)
    
    # Handle the date column in train data
    if '' in train_df.columns:  # Empty column name is the date
        train_df.rename(columns={'': 'date'}, inplace=True)
    
    # Add ID column to train data if missing
    if 'ID' not in train_df.columns:
        train_df['ID'] = range(len(train_df))
    
    print("\nSaving as parquet files...")
    train_df.to_parquet('train.parquet', index=False)
    test_df.to_parquet('test.parquet', index=False)
    
    print("\nData preparation complete!")
    print(f"Train columns: {train_df.columns.tolist()[:10]}... (showing first 10)")
    print(f"Test columns: {test_df.columns.tolist()[:10]}... (showing first 10)")
    
    # Display some statistics
    print("\nData statistics:")
    print(f"Train - Close price range: {train_df['close'].min():.2f} - {train_df['close'].max():.2f}")
    print(f"Test - Close price range: {test_df['close'].min():.2f} - {test_df['close'].max():.2f}")
    print(f"Train - Label (return) mean: {train_df['label'].mean():.6f}, std: {train_df['label'].std():.6f}")
    print(f"Test - Label (return) mean: {test_df['label'].mean():.6f}, std: {test_df['label'].std():.6f}")
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_data()
