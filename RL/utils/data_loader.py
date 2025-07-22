import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

class CryptoDataLoader:
    """
    Efficient data loader for large parquet files
    Handles streaming and batching
    """
    
    def __init__(self, 
                 file_path: str,
                 batch_size: int = 10000,
                 feature_columns: Optional[List[str]] = None):
        
        self.file_path = file_path
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.logger = logging.getLogger(__name__)
        
        # Get file info
        self.parquet_file = pq.ParquetFile(file_path)
        self.num_rows = self.parquet_file.metadata.num_rows
        self.schema = self.parquet_file.schema
        
        self.logger.info(f"Loaded parquet file with {self.num_rows} rows")
        
    def iterate_batches(self):
        """Iterate through file in batches"""
        
        for batch in self.parquet_file.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            
            if self.feature_columns:
                # Ensure all required columns exist
                missing_cols = set(self.feature_columns) - set(df.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns: {missing_cols}")
                
                # Select only specified columns
                available_cols = [col for col in self.feature_columns if col in df.columns]
                df = df[available_cols]
            
            yield df
    
    def load_chunk(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load specific chunk of data"""
        
        # Calculate which row groups to read
        row_group_size = self.parquet_file.metadata.row_group(0).num_rows
        start_group = start_idx // row_group_size
        end_group = (end_idx - 1) // row_group_size + 1
        
        # Read row groups
        table = self.parquet_file.read_row_groups(
            list(range(start_group, min(end_group, self.parquet_file.num_row_groups)))
        )
        
        df = table.to_pandas()
        
        # Slice to exact range
        relative_start = start_idx - start_group * row_group_size
        relative_end = relative_start + (end_idx - start_idx)
        
        return df.iloc[relative_start:relative_end]
    
    def load_for_walk_forward(self, window_size: int, step_size: int):
        """Load data in windows for walk-forward validation"""
        
        start = 0
        while start + window_size <= self.num_rows:
            df = self.load_chunk(start, start + window_size)
            yield df
            start += step_size
