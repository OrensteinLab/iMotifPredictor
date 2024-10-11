import pandas as pd
import numpy as np

def create_negative_bed(original_bed_path, output_bed_path):
    # Load the positive BED file into a DataFrame
    positive_df = pd.read_csv(original_bed_path, sep='\t', header=None, usecols=[0, 1, 2])

    # Remove or shorten the "chr" prefix from the first column
    positive_df[0] = positive_df[0].str.replace('chr', '')

    # Sort the DataFrame by chromosome, start, and end columns
    positive_df = positive_df.sort_values(by=[0, 1, 2])

    # Set the length of each sequence to 124 bases
    positive_df[2] = positive_df[1] + 124

    # Create an empty DataFrame with the same columns
    genome_size = positive_df.groupby(0)[2].max().reset_index()
    genome_size[1] = 0

    # Exclude regions from the positive BED
    new_df = pd.concat([genome_size, positive_df]).sort_values(by=[0, 1, 2])
    new_df = new_df.drop_duplicates(subset=[0, 1, 2], keep=False)

    # Save the new BED file
    new_df.to_csv(output_bed_path, sep='\t', header=False, index=False)

def generate_negative_coordinates(positive_bed_path, output_bed_path):
    # Load positive BED file
    positive_df = pd.read_csv(positive_bed_path, sep='\t', header=None, usecols=[0, 1, 2])

    # Remove or shorten the "chr" prefix from the first column
    positive_df[0] = positive_df[0].str.replace('chr', '')

    # Generate negative coordinates
    negative_coordinates = []
    for chrom in positive_df[0].unique():
        # Filter positive coordinates for the current chromosome
        positive_coords_chrom = positive_df[positive_df[0] == chrom][[1, 2]]

        # Generate negative coordinates by randomly sampling positions that are not covered by positive coordinates
        for _ in range(100):  # Adjust the number of negative coordinates as needed
            start = np.random.randint(0, positive_coords_chrom[2].max() - 124)
            end = start + 124
            while any((start < positive_coords_chrom[2]) & (end > positive_coords_chrom[1])):
                start = np.random.randint(0, positive_coords_chrom[2].max() - 124)
                end = start + 124
            negative_coordinates.append([chrom, start, end])

    # Create a DataFrame for negative coordinates
    negative_df = pd.DataFrame(negative_coordinates, columns=[0, 1, 2])

    # Save the negative BED file
    negative_df.to_csv(output_bed_path, sep='\t', header=False, index=False)

# Replace 'positive.bed' and 'negative.bed' with your actual file names
create_negative_bed('HEK_G4_high_confidence_peaks.bed', 'negative.bed')
generate_negative_coordinates('HEK_G4_high_confidence_peaks.bed', 'negative_coordinates.bed')
