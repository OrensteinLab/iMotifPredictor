import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)


def plot_roc_curves(files_and_models, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')  # Plot the chance line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    for file_path, label in files_and_models:
        data = pd.read_csv(file_path)
        predictions = data[data.columns[
            data.columns.str.contains('pred', case=False, regex=True)]]  # Assumes prediction columns contain 'pred'
        true_labels = data['True_Labels']
        fpr, tpr, _ = roc_curve(true_labels, predictions.iloc[:, 0])
        roc_auc = auc(fpr, tpr)

        # Define color and style based on the label
        if 'SEQ' in label:
            color = 'pink' if 'YES' in label else 'lightblue'
        else:
            color = 'grey'  # Default color if SEQ is not involved

        linestyle = '--' if 'MICRO' in label else '-'

        # Define marker based on ATAC inclusion
        marker = 'D' if 'ATAC' in label else None  # 'D' for diamond shape

        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})', color=color, linestyle=linestyle, marker=marker)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# Function to plot ROC curves
def plot_roc_curves(files_and_models):
    fig, axes = plt.subplots(nrows=1, ncols=len(files_and_models), figsize=(18, 6), sharey=True)
    fig.suptitle('ROC Curves by Negative Group and Model')

    for ax, (group, file_model_tuples) in zip(axes, files_and_models.items()):
        for file_name, model_label in file_model_tuples:
            df = load_data(file_name)
            true_labels = df['True_Labels']
            predictions = df['Predictions']
            fpr, tpr, _ = roc_curve(true_labels, predictions)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{model_label} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_title(group)
        if ax == axes[0]:
            ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'AUROC.png')
    plt.show()


# Function to plot prediction distributions
def plot_prediction_distributions(df, prediction_columns):
    sns.set(style="whitegrid")
    for prediction_column in prediction_columns:
        plt.figure(figsize=(16, 4))  # שינוי גודל התרשים לרוחב 16 וגובה 4
        for i, loop in enumerate(['Loop1_Length', 'Loop2_Length', 'Loop3_Length'], start=1):
            plt.subplot(1, 3, i)
            sns.boxplot(x=df[loop], y=df[prediction_column], palette='viridis')
            plt.title(f'{loop} - {prediction_column}')
            plt.xlabel('Loop Length', fontsize=14)
            plt.ylabel('Prediction Score', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'plot_{prediction_column}.png', dpi=300)
        plt.show()


# Function to plot heatmaps for mutation effects
def plot_and_save_heatmap_for_model(prediction_column, mutations_data):
    plt.rcParams.update({'font.size': 18})  # פונט יותר גדול
    baseline = mutations_data[mutations_data['Original_Nucleotide'] == mutations_data['Mutated_Nucleotide']]
    baseline_dict = dict(zip(baseline['Position'], baseline[prediction_column]))

    # Ensure calculation is WT - MUTATION
    def calculate_difference(row):
        return baseline_dict.get(row['Position'], row[prediction_column]) - row[prediction_column]

    mutations_data['Difference'] = mutations_data.apply(calculate_difference, axis=1)
    differences = mutations_data[mutations_data['Original_Nucleotide'] != mutations_data['Mutated_Nucleotide']]
    heatmap_data = differences.pivot_table(index='Mutated_Nucleotide', columns='Position', values='Difference',
                                           aggfunc='mean').fillna(0).reindex(['A', 'C', 'G', 'T'])

    plt.figure(figsize=(20, 6))
    ax = sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Prediction score difference\n(wild type - variant)'})

    # Set the title and axis labels
    ax.set_xlabel('Position',fontsize=14)
    ax.set_ylabel('Nucleotide',fontsize=14)

    # Adding black dots for WT positions
    wt_data = mutations_data[mutations_data['Original_Nucleotide'] == mutations_data['Mutated_Nucleotide']]
    for _, row in wt_data.iterrows():
        mut_nucleotide = row['Original_Nucleotide']
        pos = row['Position']
        ax.plot(pos - 0.5, ['A', 'C', 'G', 'T'].index(mut_nucleotide) + 0.5, 'ko')  # Adjusted to plot at the center of the cells

    # Save and show the plot
    plt.savefig(f"{prediction_column}_heatmap.png")
    plt.show()
    plt.close()





def plot_roc_curves_for_group(group_name, group_data):
    plt.figure(figsize=(9, 18))  # שינוי גודל התרשים ל-9x18 אינץ'
    plt.rcParams.update({'font.size': 20})  # פונט יותר גדול

    # Ensuring group_data is a dictionary
    if not isinstance(group_data, dict):
        print(f"Error: Data for {group_name} is not formatted correctly.")
        return

    # רשימת צבעים מורחבת
    color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'purple', 'brown',
                  'pink', 'lime', 'teal', 'lavender', 'turquoise', 'tan', 'gold', 'darkgreen', 'lightblue', 'navy',
                  'coral']

    for i, (subgroup, entries) in enumerate(group_data.items()):
        ax = plt.subplot(len(group_data), 1, i + 1)  # שינוי מספר השורות והעמודות לתצורה אנכית
        ax.plot([0, 1], [0, 1], 'k--', lw=0.5)  # קו הסיכוי
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False positive rate', fontsize=20)  # שינוי לתחתית
        ax.set_ylabel('True positive rate', fontsize=20)  # שינוי לתחתית
        for j, (file_path, label) in enumerate(entries):
            data = pd.read_csv(file_path)
            prediction_column = next((col for col in data.columns if 'predict' in col.lower()), None)
            if label == 'Microarray':
                prediction_column = next((col for col in data.columns if 'signal' in col.lower()), None)
            if label == 'ATAC':
                prediction_column = next((col for col in data.columns if 'atac' in col.lower()), None)
            if label == 'IMSeeker':
                prediction_column = 'IMSeekr_Score'
                label = 'iM-Seeker'  # Update label to 'iM-seeker'
            color = color_list[j % len(color_list)]  # קבלת הצבע הבא ברשימה

            fpr, tpr, _ = roc_curve(data['True_Labels'], data[prediction_column])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{label} ({roc_auc:.3f})', color=color, lw=1)

        legend = ax.legend(loc="lower right", fancybox=True, shadow=True, ncol=1)
        legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())

        if i == 0:
            ax.text(legend_bbox.x0+0.63, legend_bbox.y1-0.07 , 'train : HEK239T\ntest : HEK239T',
                    transform=ax.transAxes, fontsize=20, ha='left', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        else:
            ax.text(legend_bbox.x0+0.63, legend_bbox.y1-0.07 , 'train : HEK239T\ntest : WDLPS',
                    transform=ax.transAxes, fontsize=20, ha='left', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        if i == 0:
            ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
        else:
            ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'ROC_Curve_{group_name}.png')
    plt.show()


# Function to plot violin plots
def plot_combined_violin_plots(data, prediction_columns, core_length_columns):
    sns.set(style="whitegrid")

    # Filter the data to include only sequences with loop lengths between 1 and 5
    filtered_data = data

    for pred_col in prediction_columns:
        plt.figure(figsize=(16, 4))  # שינוי גודל התרשים לרוחב 16 וגובה 4
        for i, core_col in enumerate(core_length_columns, start=1):
            plt.subplot(1, len(core_length_columns), i)
            sns.violinplot(x=core_col, y=pred_col, data=filtered_data, inner="box", palette="muted")
            plt.title(f'{pred_col} - {core_col}')
            plt.xlabel('Core Length', fontsize=14)
            plt.ylabel('Prediction Score', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'violin_plot_for_{pred_col.replace("/", "_")}.png', dpi=300)
        plt.show()



        # Main execution
if __name__ == "__main__":
    # Define your file names and models here

    files_and_models = {
        'RANDOM': {
            'HEK': [
                ('../AUROC/predictions_and_true_labels_seq_random.csv', 'Sequence'),
                ('../AUROC/predictions_and_true_labels_acc_random.csv', 'Sequence + ATAC'),
                ('../AUROC/predictions_and_true_labels_micro_rand.csv', 'Sequence + microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran.csv', 'Sequence + microarray+ ATAC'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran.csv', 'Microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran.csv', 'ATAC'),
                ('../im_seeker/comparison_results_hek_pos+rand.csv', 'IMSeeker')
            ],
            'WDLPS': [
                ('../AUROC/predictions_and_true_labels_seq_random_WDLPS.csv', 'Sequence'),
                ('../AUROC/predictions_and_true_labels_acc_random_WDLPS.csv', 'Sequence + ATAC'),
                ('../AUROC/predictions_and_true_labels_micro_rand_WDLPS.csv', 'Sequence + microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran_WDLPS.csv', 'Sequence + microarray + ATAC'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran_WDLPS.csv', 'Microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_ran_WDLPS.csv', 'ATAC'),
                ('../im_seeker/comparison_results_wdlps_pos+rand.csv', 'IMSeeker')

            ]
        },
        #'DISHUFFLE': {
          #  'HEK': [
           #     ('../AUROC/predictions_and_true_labels_seq_perm.csv', 'Sequence'),
            #    ('../AUROC/predictions_and_true_labels_micro_perm.csv', 'Sequence + microarray'),
             #   ('../AUROC/predictions_and_true_labels_micro_perm.csv', 'Microarray'),
              #  ('../im_seeker/comparison_results_hek_pos+perm.csv', 'IMSeeker')

           # ],
            #'WDLPS': [
             #   ('../AUROC/predictions_and_true_labels_seq_perm_WDLPS.csv', 'Sequence'),
              #  ('../AUROC/predictions_and_true_labels_micro_perm_WDLPS.csv', 'Sequence + microarray'),
               # ('../AUROC/predictions_and_true_labels_micro_perm_WDLPS.csv', 'Microarray'),
                #('../im_seeker/comparison_results_wdlps_pos+perm.csv', 'IMSeeker')

            #]
        #},
        'GENNULSEQ': {
            'HEK': [
                ('../AUROC/predictions_and_true_labels_seq_gen.csv', 'Sequence'),
                ('../AUROC/predictions_and_true_labels_acc_gen.csv', 'Sequence + ATAC'),
                ('../AUROC/predictions_and_true_labels_micro_gen.csv', 'Sequence + microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen.csv', 'Sequence + microarray + ATAC'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen.csv', 'Microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen.csv', 'ATAC'),
                ('../im_seeker/comparison_results_hek_pos+gen.csv', 'IMSeeker')

            ],
            'WDLPS': [
                ('../AUROC/predictions_and_true_labels_seq_gen_WDLPS.csv', 'Sequence'),
                ('../AUROC/predictions_and_true_labels_acc_gen_WDLPS.csv', 'Sequence + ATAC'),
                ('../AUROC/predictions_and_true_labels_micro_gen_WDLPS.csv', 'Sequence + microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen_WDLPS.csv', 'Sequence + microarray + ATAC'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen_WDLPS.csv', 'Microarray'),
                ('../AUROC/predictions_and_true_labels_acc_mic_gen_WDLPS.csv', 'ATAC'),
                ('../im_seeker/comparison_results_wdlps_pos+gen.csv', 'IMSeeker')

            ]
        }
    }

    # Generate ROC curves for each group
    for group_name, group_data in files_and_models.items():
        plot_roc_curves_for_group(group_name, group_data)
    """"
    df = load_data('../interpation_file/merged_file.csv')
    prediction_columns = [
    'Prediction_gen',
    'Prediction_perm',
    'Prediction_rand',
    'signal'
    ]

    #plot_prediction_distributions(df, prediction_columns)
    core_length_columns = ['Core1_Length', 'Core2_Length', 'Core3_Length', 'Core4_Length']
    loop_length_columns = ['Loop1_Length', 'Loop2_Length', 'Loop3_Length']  # Adjust loop length columns as necessary
    plot_prediction_distributions(df, prediction_columns)
    # Generate violin plots
    plot_combined_violin_plots(df, prediction_columns, core_length_columns)
    
    
    mutations_data = load_data('../interpation_file/mutations.csv')
    prediction_models = [
        'Prediction_rand',
        'Prediction_dishuffle',
        'Prediction_gen',
        'Signal',
    ]
    for model in prediction_models:
        plot_and_save_heatmap_for_model(model, mutations_data)
  """