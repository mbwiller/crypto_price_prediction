import pandas as pd
import pyarrow.parquet as pq
import os

def split_parquet_file(input_file, chunk_size_mb=500):
    """
    Split a large Parquet file into smaller chunks by reading in batches.
    
    Args:
        input_file: Path to the input Parquet file
        chunk_size_mb: Maximum size of each chunk in MB (default: 500)
    """
    # Get file info
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    file_size = os.path.getsize(input_file)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    print(f"File: {input_file}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Total rows: {total_rows:,}")
    
    # Estimate rows per 500MB chunk
    # Since file is ~3.1GB and we want 500MB chunks, we need ~7 chunks
    num_chunks = int(file_size / (chunk_size_mb * 1024 * 1024)) + 1
    rows_per_chunk = total_rows // num_chunks
    
    print(f"Creating approximately {num_chunks} chunks")
    print(f"Rows per chunk: ~{rows_per_chunk:,}")
    
    # Get base filename
    base_name = os.path.splitext(input_file)[0]
    
    # Read and split the file
    chunk_num = 0
    
    # Use an iterator to read the parquet file in chunks
    for i in range(0, total_rows, rows_per_chunk):
        chunk_num += 1
        end_row = min(i + rows_per_chunk, total_rows)
        
        print(f"\nReading chunk {chunk_num}: rows {i:,} to {end_row-1:,}")
        
        # Read the entire parquet file and slice the rows we need
        # This is not ideal for memory, but works with single row group files
        if chunk_num == 1:
            # First chunk - read entire file
            df_full = pd.read_parquet(input_file)
            df_chunk = df_full.iloc[i:end_row]
        else:
            # Subsequent chunks - use already loaded data
            df_chunk = df_full.iloc[i:end_row]
        
        # Write chunk
        output_file = f"{base_name}_chunk_{chunk_num:03d}.parquet"
        df_chunk.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        # Get actual size
        actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Written: {output_file}")
        print(f"Size: {actual_size_mb:.1f} MB")
        print(f"Rows: {len(df_chunk):,}")
        
        # Free memory if this is the last chunk
        if end_row >= total_rows:
            del df_full
            del df_chunk
            break
    
    print(f"\n✓ Successfully split into {chunk_num} chunks!")

def split_parquet_file_memory_efficient(input_file, chunk_size_mb=500):
    """
    Alternative: Convert to multiple row groups first, then split.
    More memory efficient for large files.
    """
    import pyarrow as pa
    
    # Get file info
    parquet_file = pq.ParquetFile(input_file)
    schema = parquet_file.schema_arrow
    total_rows = parquet_file.metadata.num_rows
    file_size = os.path.getsize(input_file)
    
    print(f"File: {input_file}")
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Total rows: {total_rows:,}")
    
    # Calculate target rows per chunk
    num_chunks = int(file_size / (chunk_size_mb * 1024 * 1024)) + 1
    rows_per_chunk = total_rows // num_chunks
    
    print(f"Target chunks: {num_chunks}")
    print(f"Rows per chunk: ~{rows_per_chunk:,}")
    
    base_name = os.path.splitext(input_file)[0]
    
    # Read the table once
    print("\nReading original file...")
    table = pq.read_table(input_file)
    
    # Split and write chunks
    for chunk_num in range(num_chunks):
        start_idx = chunk_num * rows_per_chunk
        end_idx = min((chunk_num + 1) * rows_per_chunk, total_rows)
        
        print(f"\nCreating chunk {chunk_num + 1}: rows {start_idx:,} to {end_idx-1:,}")
        
        # Slice the table
        chunk_table = table.slice(start_idx, end_idx - start_idx)
        
        # Write chunk
        output_file = f"{base_name}_chunk_{chunk_num + 1:03d}.parquet"
        pq.write_table(chunk_table, output_file, compression='snappy')
        
        actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Written: {output_file} ({actual_size_mb:.1f} MB)")
    
    print(f"\n✓ Successfully split into {num_chunks} chunks!")

# Run the split
if __name__ == "__main__":
    input_file = r"C:\Users\Matt Willer\Downloads\drw-crypto-market-prediction\train.parquet"
    
    # Use the memory-efficient method
    split_parquet_file_memory_efficient(input_file, chunk_size_mb=500)