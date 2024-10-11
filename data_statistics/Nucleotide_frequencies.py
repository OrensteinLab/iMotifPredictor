import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

# Increase font size for the entire plot
plt.rcParams.update({
    'font.size': 24,            # General font size
    'axes.titlesize': 24,       # Title font size
    'axes.labelsize': 24,       # Axes labels font size
    'xtick.labelsize': 24,      # X-tick labels font size
    'ytick.labelsize': 24,      # Y-tick labels font size
    'legend.fontsize': 24,      # Legend font size
    'figure.titlesize': 24      # Figure title font size
})

# Function to read sequences from the provided file
def read_lengths(file_path):
    lengths = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = int(parts[1])
                end = int(parts[2])
                lengths.append(end - start)
    return lengths

# Paths to the files
file_path_hek = '../pos_bed_files/HEK_iM_high_confidence_peaks.bed'
file_path_wdlps = '../pos_bed_files/WDLPS_iM_high_confidence_peaks.bed'

# Read sequences from files
lengths_hek = read_lengths(file_path_hek)
lengths_wdlps = read_lengths(file_path_wdlps)

# Calculate lengths
median_length_hek = np.median(lengths_hek)
median_length_wdlps = np.median(lengths_wdlps)

# Create a figure for both plots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))

# Plotting the histogram with more bins and proper x-axis label
ax1.hist(lengths_hek, bins=200, color='skyblue', range=(0, 800), alpha=0.5, label=f'HEK293T (n={len(lengths_hek)})')
ax1.hist(lengths_wdlps, bins=200, color='orange', range=(0, 800), alpha=0.5, label=f'WDLPS (n={len(lengths_wdlps)})')
ax1.axvline(median_length_hek, color='blue', linestyle='dashed', linewidth=1, label=f'Median: HEK293T')
ax1.axvline(median_length_wdlps, color='darkorange', linestyle='dashed', linewidth=1, label=f'Median: WDLPS')
ax1.legend(loc='upper right')
ax1.set_xlabel('i-motif length (base pairs)')
ax1.set_ylabel('Count')
ax1.set_ylim(0, 600)
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

# Function to create a list of sequences
def createlist(file, process_extended_logic=False):
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    seq_list = []
    with open(file, 'r') as file:
        data = file.read()
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        if process_extended_logic:
            midpoint = len(sequence) // 2
            extracted_sequence = sequence[midpoint - 62:midpoint + 62]
            if len(extracted_sequence) == 124:
                complement = calculate_reverse_complement(extracted_sequence)
                c_count_sequence = extracted_sequence.count('C')
                c_count_complement = complement.count('C')
                if c_count_sequence >= c_count_complement:
                    seq_list.append(extracted_sequence)
                else:
                    seq_list.append(complement)
        else:
            seq_list.append(sequence)

    return seq_list

# Function to calculate nucleotide frequencies
def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]
    return ''.join(complement_dict[base] for base in reversed_sequence)

def calculate_frequencies(seq_list):
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for seq in seq_list:
        for nucleotide in nucleotide_counts.keys():
            nucleotide_counts[nucleotide] += seq.count(nucleotide)
    total_nucleotides = sum(nucleotide_counts.values())
    return {nt: count / total_nucleotides for nt, count in nucleotide_counts.items()}

# Assuming you have a file for each condition
files = {
    'WDLPS_Positive': '../pos_txt_files/WDLPS_iM.txt',
    'WDLPS_RandomNegative': '../random_neg/WDLPS_iM_neg.txt',
    'WDLPS_NegativePermutation': '../txt_permutaion/WDLPS_iM_perm_neg.txt',
    'WDLPS_NegativeGenNullSeq': '../genNellSeq/negWDLPSiM.txt',
    'HEK_Positive': '../pos_txt_files/HEK_iM.txt',
    'HEK_RandomNegative': '../random_neg/HEK_iM_neg.txt',
    'HEK_NegativePermutation': '../txt_permutaion/HEK_iM_perm_neg.txt',
    'HEK_NegativeGenNullSeq': '../genNellSeq/negHekiM.txt'
}

# Calculate sequences, process extended logic only for positive datasets
seq_lists = {group: createlist(file_path, 'Positive' in group) for group, file_path in files.items()}

# Calculate frequencies for each group
frequencies = {group: calculate_frequencies(seq_list) for group, seq_list in seq_lists.items()}

# Convert the frequencies to a DataFrame for easier plotting
df_frequencies = pd.DataFrame(frequencies).T

# Prepare data for plotting
# Extracting values for A, C, G, T for each group and condition
A_values = df_frequencies['A'].values
C_values = df_frequencies['C'].values
G_values = df_frequencies['G'].values
T_values = df_frequencies['T'].values

# Define the categories and groups for labeling
groups = ['WDLPS', 'HEK293T']
subcategories = ['Positive', 'Random genome', 'Dinucleotide shuffeled', 'GenNullSeqs']

# Define the y positions to ensure correct gaps
y_positions = [0, 1, 2, 3, 5, 6, 7, 8]  # Notice the gap between the 3rd and 5th positions

# Define colors for each nucleotide type
colors = ['#a1c9f4', '#ffb3e6', '#c2c2f0', '#fdae61']  # Colors for A, C, G, T

# Plot the bars for each nucleotide
for idx, (a, c, g, t) in zip(y_positions, zip(A_values, C_values, G_values, T_values)):
    values = [a, c, g, t]
    left = 0  # Starting point for each group's set of bars
    for value, color in zip(values, colors):
        ax2.barh(idx, value, left=left, color=color, edgecolor='none', height=1.0)  # Use height=1.0 for no gaps within groups
        # Centering the numeric values on each bar segment
        ax2.text(left + value / 2, idx, f'{value:.3f}', ha='center', va='center', fontsize=20)
        left += value

# Set the y-tick labels
ax2.set_yticks(y_positions)
ax2.set_yticklabels((subcategories * 2), fontsize=24)  # Repeat labels appropriately

# Position the group labels further to the right
ax2.text(1.3, 1.0, 'HEK293T', ha='right', va='center', fontsize=24, color='black', transform=ax2.transData)
ax2.text(1.3, 6.7, 'WDLPS', ha='right', va='center', fontsize=24, color='black', transform=ax2.transData)

# Set labels for axes
ax2.set_xlabel('Nucleotide frequency', fontsize=24)

# Create and place the legend outside the plot area
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
ax2.legend(legend_elements, ['A', 'C', 'G', 'T'], title="Nucleotides", bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)

# Remove spines and ticks
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)

ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

# Adjust layout
plt.tight_layout()
plt.savefig('data_stat.png')
plt.close()
