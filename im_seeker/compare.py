"""
import pandas as pd
import re


def fasta_to_dict(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as file:
        header = None
        sequence = []
        for line in file:
            if line.startswith('>'):
                if header is not None:
                    sequences[header] = ''.join(sequence)
                header = line.strip()
                sequence = []
            else:
                sequence.append(line.strip())
        if header is not None:
            sequences[header] = ''.join(sequence)
    return sequences


def process_imseeker_results(imseeker_file):
    imseeker_results = pd.read_csv(imseeker_file)
    imseeker_results[['chr', 'positions']] = imseeker_results['chr'].str.split(':', expand=True)
    imseeker_results[['beg', 'end']] = imseeker_results['positions'].str.split('-', expand=True)
    imseeker_results['beg'] = imseeker_results['beg'].astype(int)
    imseeker_results['end'] = imseeker_results['end'].astype(int)
    return imseeker_results


def compare_sequences(fasta_file, imseeker_file, output_file,true_label):
    fasta_sequences = fasta_to_dict(fasta_file)
    imseeker_results = process_imseeker_results(imseeker_file)

    with open(output_file, 'w') as outfile:
        outfile.write("True_Label,IMSeekr_Score\n")
        for header, sequence in fasta_sequences.items():
            imseeker_score = 0
            chrom, pos = header[1:].split(':')
            start, end = map(int, pos.split('-'))
            matched = imseeker_results[
                (imseeker_results['chr'] == chrom) &
                (imseeker_results['beg'] == start) &
                (imseeker_results['end'] == end)
                ]
            if not matched.empty:
                imseeker_score = matched.iloc[0]['predict_score']
            outfile.write(f"{true_label},{imseeker_score}\n")


# הפעלת הפונקציה עם הקבצים שהעלית

compare_sequences('../fasta_files/HEK_iM_chr1.fasta', 'HEK_im_pred.csv', 'comparison_results_hek_pos.csv', 1)
compare_sequences('../fasta_files/HEK_iM_neg_chr1.fasta', 'HEK_rand.csv', 'comparison_results_hek_rand.csv',0)
compare_sequences('../fasta_files/negHekiM_chr1.fasta', 'hek_im_gen.csv', 'comparison_results_hek_gen.csv', 0)
compare_sequences('../fasta_files/HEK_iM_perm_neg_chr1.fasta', 'hek_im_perm.csv', 'comparison_results_hek_perm.csv', 0)
compare_sequences('../fasta_files/WDLPS_iM.fasta', 'WDLPS_im.csv', 'comparison_results_wdlps_pos.csv', 1)
compare_sequences('../fasta_files/WDLPS_iM_neg.fasta', 'WDLPS_rand.csv', 'comparison_results_wdlps_rand.csv', 0)
compare_sequences('../fasta_files/WDLPS_iM_perm_neg.fasta', 'WDLPS_perm.csv', 'comparison_results_wdlps_perm.csv', 0)
compare_sequences('../fasta_files/negWDLPSiM.fasta', 'WDLPS_gen.csv', 'comparison_results_wdlps_gen.csv', 0)
"""

import pandas as pd


def merge_csv_files(file1, file2, output_file):
    # קרא את הקבצים
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # איחוד הנתונים
    all_data = pd.concat([data1, data2])

    # שמירת הנתונים המאוחדים לקובץ חדש
    all_data.to_csv(output_file, index=False)
    print(f"איחוד הנתונים הושלם ונשמר בקובץ: {output_file}")


# דוגמה לשימוש בפונקציה
merge_csv_files('comparison_results_hek_pos.csv', 'comparison_results_hek_rand.csv',
                'comparison_results_hek_pos+rand.csv')
import pandas as pd


def merge_csv_files(file1, file2, output_file):
    # קרא את הקבצים
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # איחוד הנתונים
    all_data = pd.concat([data1, data2])
    if 'True_Label' in all_data.columns:
        all_data.rename(columns={'True_Label': 'True_Labels'}, inplace=True)

    all_data = all_data.sample(frac=1).reset_index(drop=True)

    # שמירת הנתונים המאוחדים לקובץ חדש
    all_data.to_csv(output_file, index=False)
    print(f"איחוד הנתונים הושלם ונשמר בקובץ: {output_file}")


# דוגמה לשימוש בפונקציה
merge_csv_files('comparison_results_hek_pos.csv', 'comparison_results_hek_rand.csv','comparison_results_hek_pos+rand.csv')
merge_csv_files('comparison_results_hek_pos.csv', 'comparison_results_hek_perm.csv','comparison_results_hek_pos+perm.csv')
merge_csv_files('comparison_results_hek_pos.csv', 'comparison_results_hek_gen.csv','comparison_results_hek_pos+gen.csv')
merge_csv_files('comparison_results_wdlps_pos.csv', 'comparison_results_wdlps_rand.csv','comparison_results_wdlps_pos+rand.csv')
merge_csv_files('comparison_results_wdlps_pos.csv', 'comparison_results_wdlps_perm.csv','comparison_results_wdlps_pos+perm.csv')
merge_csv_files('comparison_results_wdlps_pos.csv', 'comparison_results_wdlps_gen.csv','comparison_results_wdlps_pos+gen.csv')




