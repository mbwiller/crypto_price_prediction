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
    print(f"\nTrain label stats: mean={train_df['label'].mean():.6f}, std={train_df['label'].std():.6f}")
    print(f"Test label stats: mean={test_df['label'].mean():.6f}, std={test_df['label'].std():.6f}")
    
    # Since we don't have a 'close' column, we need to create one
    # The label appears to be the price return, so we'll create prices from it
    print("\nCreating synthetic close prices...")
    
    # For training data
    if 'close' not in train_df.columns:
        # Start with a base price (e.g., 50000 for Bitcoin-like prices)
        base_price = 50000.0
        
        # Initialize close prices array
        close_prices = np.zeros(len(train_df))
        close_prices[0] = base_price
        
        # Generate prices using the label as returns
        # If label is already a return (price change), use it directly
        # If label seems to be in a different scale, adjust accordingly
        for i in range(1, len(train_df)):
            # Assuming label is the return (percentage change)
            # Clip extreme values to prevent numerical issues
            return_val = np.clip(train_df['label'].iloc[i-1], -0.1, 0.1)  # Max 10% change per minute
            close_prices[i] = close_prices[i-1] * (1 + return_val)
        
        train_df['close'] = close_prices
    
    # For test data
    if 'close' not in test_df.columns:
        # If test labels are all zero, we need a different approach
        if test_df['label'].std() == 0:
            print("Warning: Test labels are all zeros. Creating synthetic price movements.")
            # Create small random walk for testing
            last_train_price = train_df['close'].iloc[-1] if 'close' in train_df.columns else 50000.0
            
            close_prices = np.zeros(len(test_df))
            close_prices[0] = last_train_price
            
            # Generate small random returns for testing
            for i in range(1, len(test_df)):
                random_return = np.random.normal(0, 0.001)  # 0.1% std dev
                close_prices[i] = close_prices[i-1] * (1 + random_return)
            
            test_df['close'] = close_prices
        else:
            # Use labels if they're not all zero
            last_train_price = train_df['close'].iloc[-1] if 'close' in train_df.columns else 50000.0
            close_prices = np.zeros(len(test_df))
            close_prices[0] = last_train_price
            
            for i in range(1, len(test_df)):
                return_val = np.clip(test_df['label'].iloc[i-1], -0.1, 0.1)
                close_prices[i] = close_prices[i-1] * (1 + return_val)
            
            test_df['close'] = close_prices
    
    # Add high/low columns (required for some technical indicators)
    for df in [train_df, test_df]:
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.002  # 0.2% above close
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.998   # 0.2% below close
    
    # Ensure all required columns have proper data types
    float_columns = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'close', 'high', 'low', 'label']
    float_columns.extend([f'X{i}' for i in range(1, 781)])
    
    for col in float_columns:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0)
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0)
    
    # Handle the date column in train data
    if 'Unnamed: 0' in train_df.columns:
        train_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    elif '' in train_df.columns:
        train_df.rename(columns={'': 'date'}, inplace=True)
    
    # Add ID column to train data if missing
    if 'ID' not in train_df.columns:
        train_df['ID'] = range(len(train_df))
    
    # Fix label column for proper training
    # Recalculate labels as price returns
    for df in [train_df, test_df]:
        df['label'] = df['close'].pct_change().fillna(0)
        # Clip extreme values
        df['label'] = np.clip(df['label'], -0.1, 0.1)
    
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
    
    # Check for any issues
    print("\nData validation:")
    print(f"Train NaN values: {train_df.isna().sum().sum()}")
    print(f"Test NaN values: {test_df.isna().sum().sum()}")
    print(f"Train infinite values: {np.isinf(train_df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Test infinite values: {np.isinf(test_df.select_dtypes(include=[np.number])).sum().sum()}")
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_data()
