import pandas as pd

# Original sequence (example sequence here, replace with your actual sequence)
original_sequence = "TATGAATAAACTCCCATTGCATTTGTTGGGGGGAGTCATGTATTATGCATTATGTATGCACAGCTATCTGGATTGGATACCTTCCACCCAGACTGAGTCCCCCAATTTGCTGCCAAAGCAGCAG"
# Possible nucleotides
nucleotides = ['A', 'C', 'G', 'T']

# Function to generate all possible single-nucleotide mutations for a given sequence
def generate_mutations(sequence):
    mutations = []
    for i, original_nucleotide in enumerate(sequence):
        for nucleotide in nucleotides:
                # Generate a mutated version of the sequence
                mutated_sequence = sequence[:i] + nucleotide + sequence[i+1:]
                mutations.append((i+1, original_nucleotide, nucleotide, mutated_sequence))
    return mutations

upper_seq=original_sequence.upper()
midpoint = len(upper_seq) // 2
# Extract 62 bases to the right and 62 bases to the left from the midpoint
extracted_sequence = upper_seq[midpoint - 62:midpoint + 62]
# Generate mutations
mutations = generate_mutations(extracted_sequence)

# Create a DataFrame and save to CSV
df = pd.DataFrame(mutations, columns=['Position', 'Original_Nucleotide', 'Mutated_Nucleotide', 'Mutated_Sequence'])
df.to_csv('mutations.csv', index=False)

print("Mutations generated and saved to utations.csv.")
