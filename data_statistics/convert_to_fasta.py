import re

def process_and_filter_sequences(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()

    # Extract all sequences using regex
    pattern = r'(chr\d+:\d+-\d+)\n([ATCGatcg\n]+)'
    matches = re.findall(pattern, data)

    with open(output_file, 'w') as outfile:
        for match in matches:
            chromosome, sequence = match
            sequence = sequence.replace('\n', '').upper()
            midpoint = len(sequence) // 2
            # Extract 62 bases to the right and 62 bases to the left from the midpoint
            extracted_sequence = sequence[midpoint - 62:midpoint + 62]
            if len(extracted_sequence) == 124:
                chrom_parts = chromosome.split(':')
                start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
                end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
                new_header = f'>{chrom_parts[0]}:{start_pos}-{end_pos}'
                outfile.write(f"{new_header}\n")
                outfile.write(f"{extracted_sequence}\n")


"""
# Convert the files to FASTA format
process_and_filter_sequences('../pos_txt_files/WDLPS_iM.txt', '../fasta_files/WDLPS_iM.fasta')
process_and_filter_sequences('../pos_txt_files/HEK_iM.txt', '../fasta_files/HEK_iM.fasta')
process_and_filter_sequences('../genNellSeq/negHekiM.txt', '../fasta_files/negHekiM.fasta')
process_and_filter_sequences('../genNellSeq/negWDLPSiM.txt', '../fasta_files/negWDLPSiM.fasta')
process_and_filter_sequences('../random_neg/HEK_iM_neg.txt', '../fasta_files/HEK_iM_neg.fasta')
process_and_filter_sequences('../random_neg/WDLPS_iM_neg.txt', '../fasta_files/WDLPS_iM_neg.fasta')
process_and_filter_sequences('../txt_permutaion/HEK_iM_perm_neg.txt', '../fasta_files/HEK_iM_perm_neg.fasta')
process_and_filter_sequences('../txt_permutaion/WDLPS_iM_perm_neg.txt', '../fasta_files/WDLPS_iM_perm_neg.fasta')
"""
import pandas as pd
import re




# סינון לפי כרומוזום 1






def filter_chromosome_directly(input_file, output_file, chromosome='chr1'):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        write = False
        for line in infile:
            if line.startswith('>'):
                write = (line.split(':')[0] == f'>{chromosome}')
                if write:
                    outfile.write(line)
            elif write:
                outfile.write(line)


# שימוש בפונקציה כדי לסנן את כרומוזום 1
filter_chromosome_directly('../fasta_files/HEK_iM_neg.fasta', '../fasta_files/HEK_iM_neg_chr1.fasta')
#filter_chromosome_directly('../fasta_files/negHekiM.fasta', '../fasta_files/negHekiM_chr1.fasta')
#filter_chromosome_directly('../fasta_files/HEK_iM_perm_neg.fasta', '../fasta_files/HEK_iM_perm_neg_chr1.fasta')
#filter_chromosome_directly('../fasta_files/HEK_iM.fasta', '../fasta_files/HEK_iM_chr1.fasta')

print("סינון רצפי כרומוזום 1 הושלם.")





print("Conversion to FASTA format completed.")
