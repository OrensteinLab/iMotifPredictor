"""

import pandas as pd

# Load the original CSV file
original_df = pd.read_csv('../microarray_files/final_table_microarray.csv')

# Load the additional CSV file
additional_df = pd.read_csv('microarray_prediction.csv')
additional_df.rename(columns={'chr': 'New_ID'}, inplace=True)

# Merge the two dataframes based on the New_ID column
merged_df = original_df.merge(additional_df[['New_ID', 'predict_score']], on='New_ID', how='left')

# Create the new imseeker column
merged_df['imseeker'] = merged_df['predict_score'].fillna(0)

# Select the required columns
final_df = merged_df[['New_ID', 'iMab100nM_6.5_5', 'imseeker']]

# Save the final dataframe to a new CSV file
final_df.to_csv('imseeker_and_microarray.csv', index=False)
"""

import pandas as pd
from scipy.stats import pearsonr

# Load the merged dataframe from the CSV file
final_df = pd.read_csv('test_table_microarray.csv')

# Calculate Pearson correlation and p-value
correlation, p_value = pearsonr(final_df['iMab100nM_6.5_5'], final_df['imseeker'])

print("Pearson Correlation:", correlation)
print("P-value:", p_value)
""""
import pandas as pd

# קריאת קובץ ה-IDs
test_ids_file = '../test_ids.txt'
with open(test_ids_file, 'r') as file:
    test_ids = file.read().splitlines()

# קריאת הטבלה המקורית
original_file = 'imseeker_and_microarray.csv'  # יש להתאים את הנתיב לקובץ שלך
df = pd.read_csv(original_file)

# סינון הטבלה לפי ה-IDs
filtered_df = df[df['New_ID'].isin(test_ids)]

# שמירת הטבלה המסוננת לקובץ חדש
filtered_file = 'test_table_microarray.csv'
filtered_df.to_csv(filtered_file, index=False)

print(f'The filtered table has been saved to {filtered_file}')
"""