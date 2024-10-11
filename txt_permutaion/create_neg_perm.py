import ushuffle
import re

def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def create_neg_from_positive(positive_file, negative_file):
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    with open(positive_file, 'r') as file:
        data = file.read()

    negative_dict = {}
    dict = {}
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = extracted_sequence.count('C')
            c_count_complement = complement.count('C')
            if c_count_sequence >= c_count_complement:
                dict[extracted_sequence] = chromosome
            else:
                dict[complement] = chromosome

    k = 2  # k-let size
    for sequence, chromosome in dict.items():
        shuffled_sequence = ushuffle.shuffle(sequence.encode('utf-8'), k).decode('utf-8')
        negative_dict[shuffled_sequence] = chromosome

    with open(negative_file, 'w') as file:
        for sequence, chromosome in negative_dict.items():
            file.write(f">{chromosome}\n{sequence}\n")

    return negative_dict

# Paths to input and output files
positive_file = '../pos_txt_files/HEK_iM.txt'
negative_file = 'HEK_iM_perm_neg.txt'
negatives_dict = create_neg_from_positive(positive_file, negative_file)
