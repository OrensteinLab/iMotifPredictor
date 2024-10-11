import pandas as pd
import numpy as np

# Load the BedGraph file
bedgraph_path = 'coverage_profile.bedgraph'
bedgraph_df = pd.read_csv(bedgraph_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'score'])

# Filter the DataFrame for chromosome 1
bedgraph_df = bedgraph_df[bedgraph_df['chrom'] == 'chr1']

# Initialize an array to hold the ATAC signal scores for the first 25 million nucleotides
atac_signals = np.zeros(25000000)

# Iterate through the filtered bedgraph DataFrame to update the ATAC signal scores
for _, row in bedgraph_df.iterrows():
    if row['start'] < 25000000:
        end_pos = min(row['end'], 25000000)  # Ensure not to exceed the 25 million boundary
        start_pos = row['start'] - 1  # Convert to zero-based indexing
        segment_length = end_pos - start_pos
        # Adjust the score to be relative to the segment length
        relative_score = row['score'] / segment_length
        atac_signals[start_pos:end_pos] += relative_score

# Initialize an array for window scores, considering the window of 124 nucleotides
window_scores = np.zeros(len(atac_signals) - 123)

# Calculate the ATAC signal for each window
for i in range(len(window_scores)):
    window_scores[i] = np.mean(atac_signals[i:i+124])  # Calculate the relative signal

# Prepare the DataFrame for saving to CSV
windows_df = pd.DataFrame({
    'Window Start': np.arange(1, len(window_scores) + 1),
    'Window End': np.arange(124, len(window_scores) + 124),
    'ATAC Signal': window_scores
})

# Specify the path for the CSV file to save the results
csv_path = '/mnt/data/atac_signal_windows.csv'
windows_df.to_csv(csv_path, index=False)

print(f"Results saved to: {csv_path}")
