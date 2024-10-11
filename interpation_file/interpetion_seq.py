import csv
from itertools import product

# Define the parameters for the sequence generation
core_lengths = range(2, 11)  # 'C' can be from 2 to 10 in length
loop_lengths = range(1, 13)  # Loop lengths can vary from 1 to 12 nucleotides
sequence_length = 124


# Define function to create sequence
def create_sequence(core1, loop1, core2, loop2, core3, loop3, core4):
    # Create the loops based on the current combination
    sequence = ('C' * core1 + 'N' * loop1 +
                'C' * core2 + 'N' * loop2 +
                'C' * core3 + 'N' * loop3 +
                'C' * core4)

    # Pad the sequence with 'N' to achieve the desired sequence length
    padding_length = sequence_length - len(sequence)
    padded_sequence = 'N' * (padding_length // 2) + sequence + 'N' * (padding_length // 2)
    return padded_sequence


# Open a file to write the sequences
csv_filename = 'central_imotif_sequences_varying_c.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(
        ['Sequence', 'Core1_Length', 'Loop1_Length', 'Core2_Length', 'Loop2_Length', 'Core3_Length', 'Loop3_Length',
         'Core4_Length'])

    # Generate sequences for all combinations of core and loop lengths
    for core1, core2, core3, core4 in product(core_lengths, repeat=4):
        for loop1, loop2, loop3 in product(loop_lengths, repeat=3):
            sequence = create_sequence(core1, loop1, core2, loop2, core3, loop3, core4)
            if len(sequence) == sequence_length:
                # Write the sequence and its characteristics to the CSV file
                writer.writerow([sequence, core1, loop1, core2, loop2, core3, loop3, core4])

# Output a confirmation with the filename
print(f'Sequences have been written to {csv_filename}')
