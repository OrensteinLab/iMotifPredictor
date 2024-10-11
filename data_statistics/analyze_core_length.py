import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# קריאת הקובץ לקובץ DataFrame
file_path = '../interpation_file/merged_file.csv'
data = pd.read_csv(file_path)

# פונקציה להגדרת טווחי ה-Loop Length
def define_loop_ranges(data):
    def loop_length_range_v2(row):
        if row['Loop1_Length'] == 1 and row['Loop2_Length'] == 1 and row['Loop3_Length'] == 1:
            return '1'
        elif 1 <= row['Loop1_Length'] <= 2 and 1 <= row['Loop2_Length'] <= 2 and 1 <= row['Loop3_Length'] <= 2:
            return '1-2'
        elif 1 <= row['Loop1_Length'] <= 3 and 1 <= row['Loop2_Length'] <= 3 and 1 <= row['Loop3_Length'] <= 3:
            return '1-3'
        elif 2 <= row['Loop1_Length'] <= 3 and 2 <= row['Loop2_Length'] <= 3 and 2 <= row['Loop3_Length'] <= 3:
            return '2-3'
        elif 4 <= row['Loop1_Length'] <= 5 and 4 <= row['Loop2_Length'] <= 5 and 4 <= row['Loop3_Length'] <= 5:
            return '4-5'
        elif 1 <= row['Loop1_Length'] <= 5 and 1 <= row['Loop2_Length'] <= 5 and 1 <= row['Loop3_Length'] <= 5:
            return '1-5'
        elif 1 <= row['Loop1_Length'] <= 4 and 1 <= row['Loop2_Length'] <= 4 and 1 <= row['Loop3_Length'] <= 4:
            return '1-4'
        elif 4 <= row['Loop1_Length'] <= 6 and 4 <= row['Loop2_Length'] <= 6 and 4 <= row['Loop3_Length'] <= 6:
            return '4-6'
        elif 3 <= row['Loop1_Length'] <= 6 and 3 <= row['Loop2_Length'] <= 6 and 3 <= row['Loop3_Length'] <= 6:
            return '3-6'
        elif 7 <= row['Loop1_Length'] <= 9 and 7 <= row['Loop2_Length'] <= 9 and 7 <= row['Loop3_Length'] <= 9:
            return '7-9'
        elif 6 <= row['Loop1_Length'] <= 9 and 6 <= row['Loop2_Length'] <= 9 and 6 <= row['Loop3_Length'] <= 9:
            return '6-9'
        elif 0 <= row['Loop1_Length'] <= 12 and 0 <= row['Loop2_Length'] <= 12 and 0 <= row['Loop3_Length'] <= 12:
            return '0-12'
        elif 10 <= row['Loop1_Length'] <= 12 and 10 <= row['Loop2_Length'] <= 12 and 10 <= row['Loop3_Length'] <= 12:
            return '10-12'
        elif 8 <= row['Loop1_Length'] <= 10 and 8 <= row['Loop2_Length'] <= 10 and 8 <= row['Loop3_Length'] <= 10:
            return '8-10'
        elif 8 <= row['Loop1_Length'] <= 12 and 8 <= row['Loop2_Length'] <= 12 and 8 <= row['Loop3_Length'] <= 12:
            return '8-12'
        else:
            return 'Other'

    data['Loop_Range'] = data.apply(loop_length_range_v2, axis=1)
    return data

# פונקציה ליצירת ה-heatmap mutation
def create_heatmap(data):
    loop_columns = ['Loop1_Length', 'Loop2_Length', 'Loop3_Length']
    core_columns = ['Core1_Length', 'Core2_Length', 'Core3_Length', 'Core4_Length']
    columns = ['Prediction_gen', 'Prediction_perm', 'Prediction_rand', 'signal']
    plt.rcParams.update({'font.size': 18})  # פונט יותר גדול
    heatmap_data = pd.DataFrame(columns=['Loop Length', 'Core Length', 'Prediction'])


    for prediction_column in columns:
        for loop_col in loop_columns:
            for core_col in core_columns:
                temp_df = data[[loop_col, core_col, prediction_column]].rename(
                    columns={loop_col: 'Loop Length', core_col: 'Core Length', prediction_column: 'Prediction'}
                )
                heatmap_data = pd.concat([heatmap_data, temp_df])

        plt.figure(figsize=(12, 8))
        heatmap_pivot = heatmap_data.pivot_table(index='Loop Length', columns='Core Length', values='Prediction', aggfunc='mean')
        sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt='.2f')
        plt.xlabel('Core length')
        plt.ylabel('Loop length')
        plt.savefig(f'heatmap_{prediction_column}.png')
        plt.show()

# פונקציה ליצירת תרשים הקווים
def create_line_plot(data):
    melted_data = pd.melt(data, id_vars=['signal', 'Loop_Range'],
                          value_vars=['Core1_Length', 'Core2_Length', 'Core3_Length', 'Core4_Length'],
                          var_name='Core', value_name='Core_Length')

    average_signal_by_core_and_loop = melted_data.groupby(['Core_Length', 'Loop_Range'])['signal'].mean().reset_index()

    specified_ranges = ['1', '1-2', '1-3', '2-3', '4-5', '1-5', '1-4', '4-6', '3-6', '7-9', '6-9', '0-12', '10-12', '8-10', '8-12']

    plt.figure(figsize=(14, 10))

    colors = {'1': 'yellow', '1-2': 'red', '1-3': 'green', '2-3': 'blue', '4-5': 'purple',
              '1-5': 'orange', '1-4': 'pink', '4-6': 'cyan', '3-6': 'brown', '7-9': 'grey',
              '6-9': 'black', '0-12': 'magenta', '10-12': 'navy', '8-10': 'lime', '8-12': 'teal'}

    for loop_range in specified_ranges:
        subset = average_signal_by_core_and_loop[average_signal_by_core_and_loop['Loop_Range'] == loop_range]
        if not subset.empty:
            plt.plot(subset['Core_Length'], subset['signal'], marker='o', color=colors.get(loop_range, 'grey'), label=loop_range)

    plt.title('Average Microarray Signal by Core Length and Loop Length Range')
    plt.xlabel('Core Length')
    plt.ylabel('Average microarray Signal')
    plt.legend(title='Loop Length Range')
    plt.grid(True)
    plt.show()

# קריאה לפונקציות
data_with_ranges = define_loop_ranges(data)
create_heatmap(data_with_ranges)
#create_line_plot(data_with_ranges)
