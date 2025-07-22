import pandas as pd
import numpy as np

def prepare_data():
    """Prepare train and test data for the RL system"""
    
    print("Loading CSV files...")
    train_df = pd.read_csv('train_first_1000.csv')
    test_df = pd.read_csv('test_first_1000.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check if label values are reasonable
    print(f"\nOriginal train label stats: mean={train_df['label'].mean():.6f}, std={train_df['label'].std():.6f}")
    print(f"Original test label stats: mean={test_df['label'].mean():.6f}, std={test_df['label'].std():.6f}")
    
    # The labels seem to be in an unusual scale, let's normalize them
    # If mean is 0.38 and std is 0.87, these might not be returns but some other scale
    if train_df['label'].std() > 0.5:  # Unusually high
        print("\nNormalizing labels to reasonable return values...")
        # Convert to reasonable returns (target ~0.1% std dev for minute data)
        train_df['label'] = (train_df['label'] - train_df['label'].mean()) / train_df['label'].std() * 0.001
    
    print("\nCreating synthetic close prices...")
    
    # For training data
    if 'close' not in train_df.columns:
        base_price = 50000.0
        close_prices = np.zeros(len(train_df))
        close_prices[0] = base_price
        
        for i in range(1, len(train_df)):
            # Use normalized returns
            return_val = train_df['label'].iloc[i-1]
            # Additional safety: clip to reasonable minute-level returns
            return_val = np.clip(return_val, -0.05, 0.05)  # Max 5% change per minute
            close_prices[i] = close_prices[i-1] * (1 + return_val)
        
        train_df['close'] = close_prices
    
    # For test data
    if 'close' not in test_df.columns:
        # Since test labels are all zero, create small random walk
        print("Creating synthetic test prices with random walk...")
        last_train_price = train_df['close'].iloc[-1] if 'close' in train_df.columns else 50000.0
        
        close_prices = np.zeros(len(test_df))
        close_prices[0] = last_train_price
        
        # Generate small random returns for testing
        np.random.seed(42)  # For reproducibility
        for i in range(1, len(test_df)):
            random_return = np.random.normal(0, 0.001)  # 0.1% std dev
            close_prices[i] = close_prices[i-1] * (1 + random_return)
        
        test_df['close'] = close_prices
    
    # Add high/low columns
    for df in [train_df, test_df]:
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.002
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.998
    
    # Ensure all required columns have proper data types
    float_columns = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'close', 'high', 'low', 'label']
    float_columns.extend([f'X{i}' for i in range(1, 781)])
    
    for col in float_columns:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0)
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0)
    
    # Handle the date column
    if 'Unnamed: 0' in train_df.columns:
        train_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    
    # Add ID column to train data if missing
    if 'ID' not in train_df.columns:
        train_df = train_df.copy()  # Avoid fragmentation warning
        train_df['ID'] = range(len(train_df))
    
    # Recalculate labels as price returns for consistency
    for df in [train_df, test_df]:
        df['return'] = df['close'].pct_change().fillna(0)
        # Use the calculated return as the label
        df['label'] = np.clip(df['return'], -0.05, 0.05)
    
    print("\nSaving as parquet files...")
    train_df.to_parquet('train.parquet', index=False)
    test_df.to_parquet('test.parquet', index=False)
    
    print("\nData preparation complete!")
    print(f"Train columns: {train_df.columns.tolist()[:10]}... (showing first 10)")
    print(f"Test columns: {test_df.columns.tolist()[:10]}... (showing first 10)")
    
    # Display some statistics
    print("\nData statistics:")
    print(f"Train - Close price range: ${train_df['close'].min():.2f} - ${train_df['close'].max():.2f}")
    print(f"Test - Close price range: ${test_df['close'].min():.2f} - ${test_df['close'].max():.2f}")
    print(f"Train - Label (return) mean: {train_df['label'].mean():.6f}, std: {train_df['label'].std():.6f}")
    print(f"Test - Label (return) mean: {test_df['label'].mean():.6f}, std: {test_df['label'].std():.6f}")
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_data()
