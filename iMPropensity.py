import random
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import logomaker
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, gaussian_kde

# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 5
window = random.randint(5, 20)  # Adjust the range as needed
st = random.randint(1, 10)  # Adjust the range as needed
nt = random.randint(1, 10)  # Adjust the range as needed
seq_length = 124
window_seq=60

class SequenceData:
    def __init__(self, id, sequence, classification=None, accessibility=None,
                 start_coordinate=None, end_coordinate=None, micro=None):
        self.ID = id
        self.sequence = sequence.upper()  # Ensuring the sequence is in uppercase
        self.classification = classification
        self.accessibility = accessibility
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate
        self.microarray_signal = micro

    def calculate_reverse_complement(self):
        complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        reversed_sequence = self.sequence[::-1]  # Reverse the sequence
        return ''.join(complement_dict.get(base, base) for base in reversed_sequence)

    def extracted_sequence(self, length=124):
        midpoint = len(self.sequence) // 2
        cut_from_seq = length // 2
        return self.sequence[midpoint - cut_from_seq:midpoint + cut_from_seq]

################for the integrated gradients################
def integrated_gradients(inputs, model, target_class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros(inputs.shape)
    else:
        baseline = baseline

    interpolated_inputs = [
        baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)
    ]

    interpolated_inputs = [tensorflow.convert_to_tensor(input, dtype=tensorflow.float32) for input in interpolated_inputs]

    grads = []
    for input in interpolated_inputs:
        with tensorflow.GradientTape() as tape:
            tape.watch(input)
            preds = model(tensorflow.expand_dims(input, axis=0))  # Add batch dimension
            target_class = preds[:, target_class_idx]

        grads.append(tape.gradient(target_class, input))

    avg_grads = np.mean(grads, axis=0)
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads


def plot_integrated_gradients_logomaker(attributions, sequence):
    # Prepare the data for logomaker
    data = pd.DataFrame(attributions, columns=['A', 'C', 'G', 'T'])

    # Create the logo plot
    fig, ax = plt.subplots(figsize=(15, 5))
    crp_logo = logomaker.Logo(data, ax=ax, shade_below=0.5, fade_below=0.5, font_name='Arial Rounded MT Bold')

    # Style the logo plot
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.ax.set_ylabel("Attribution score", labelpad=-1, fontsize=14)
    crp_logo.ax.set_xticks([])
    crp_logo.ax.axhline(0, color='black', linewidth=0.5)
    plt.title('Integrated Gradients for Sequence')
    plt.show()

def one_hot_encode(sequence):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        one_hot = np.zeros((len(sequence), 4))
        for i, nucleotide in enumerate(sequence):
            one_hot[i, mapping[nucleotide]] = 1
        return one_hot
def integrated_gradients_example(my_model):
    sequence = "TATGAATAAACTCCCATTGCATTTGTTGGGGGGAGTCATGTATTATGCATTATGTATGCACAGCTATCTGGATTGGATACCTTCCACCCAGACTGAGTCCCCCAATTTGCTGCCAAAGCAGCAG"  # Replace with your actual sequence
    one_hot_sequence = one_hot_encode(sequence)
    target_class_idx = 0  # Change as per your target class

    attributions = np.zeros((len(sequence), 4))
    count_matrix = np.zeros(len(sequence))

    for i in range(0, len(sequence) - window_seq + 1):
        window_sequence = sequence[i:i + window_seq]
        one_hot_window = one_hot_encode(window_sequence)
        window_attributions = integrated_gradients(one_hot_window, my_model, target_class_idx)

        attributions[i:i + window_seq] += window_attributions
        count_matrix[i:i + window_seq] += 1

    # Average the attributions
    attributions /= count_matrix[:, None]

    # Aggregate attributions to match sequence length
    aggregated_attributions = np.sum(attributions, axis=1)

    # Visualize the integrated gradients
    plot_integrated_gradients_logomaker(attributions, sequence)


#####################################################################
def create_train_list(file_path):
    train_list = []
    data = pd.read_csv(file_path)

    for index, row in data.iterrows():
        seq_data = SequenceData(
            id=row['New_ID'],  # Fill in as appropriate
            sequence=row['ProbeSeq'],
            classification=None,  # Fill in as appropriate
            accessibility=None,  # Fill in as appropriate
            start_coordinate=None,  # Fill in as appropriate
            end_coordinate=None,  # Fill in as appropriate
            micro=row['iMab100nM_6.5_5']
        )
        seq_data.sequence = seq_data.extracted_sequence(window_seq)
        train_list.append(seq_data)

    return train_list

def create_test_list(file_path):
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    test_list = []

    with open(file_path, 'r') as file:
        data = file.read()
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        seq_data = SequenceData(chromosome, sequence)
        extracted_sequence = seq_data.extracted_sequence(124)

        if len(extracted_sequence) == 124:
            seq_data.sequence = extracted_sequence  # Update the sequence in seq_data
            complement = seq_data.calculate_reverse_complement()
            c_count_sequence = seq_data.sequence.count('C')
            c_count_complement = complement.count('C')

            # Select the sequence with the most G's
            chosen_sequence = extracted_sequence if c_count_sequence >= c_count_complement else complement
            seq_data.sequence = chosen_sequence  # Update the sequence in seq_data
            test_list.append(seq_data)

    return test_list



def save_signals_to_csv(sequences_test, signals, filename):
    df = pd.DataFrame({'Sequence': sequences_test, 'Signal': signals})
    df.to_csv(filename, index=False)


def append_predictions_to_csv(sequences_test, signals, filename):
    df = pd.DataFrame({'Sequence': sequences_test, 'Signal': signals})
    df.to_csv(filename, mode='a', header=not pd.read_csv(filename).empty, index=False)

def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)

    # Creating Hidden Layer
    hidden1 = Dense(fc)(pool)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer for regression
    output = Dense(1, activation='linear')(hidden1)  # Changed activation to 'linear' for regression

    # Final Model Definition
    mdl = Model(inputs=in1, outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = tensorflow.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error'])
    # Changed loss to 'mean_squared_error' for regression and added appropriate metrics

    return mdl

def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],  'N': [0.33, 0, 0.33, 0.33]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)

def load_sequences_from_csv(csv_path, nrows=None, skiprows=None):
    """Load sequences from the provided CSV file and create SequenceData objects."""
    header = pd.read_csv(csv_path, nrows=0).columns
    data = pd.read_csv(csv_path, nrows=nrows, skiprows=skiprows, names=header, header=0)
    sequences = data['Sequence'].tolist()
    return [SequenceData(chromosome=None, sequence=seq) for seq in sequences]  # Simplified object creation



def main():
    #file_test = 'genNellSeq/negWDLPSiM.txt'
    file_train = 'microarray_files/final_table_microarray.csv'
    sequences_df = pd.read_csv('interpation_file/central_imotif_sequences_varying_c.csv')
    # Create train and test lists using the respective functions
    train_list = create_train_list(file_train)
    #test_list = create_test_list(file_test)
    test_list = load_sequences_from_csv('interpation_file/central_imotif_sequences_varying_c.csv', nrows=100000,skiprows=60000)  # Load only 10 sequences for testing

    # Shuffle the training data
    random.shuffle(train_list)

    # Prepare data for training
    x_train = np.array([one_hot_encoding(seq_data.sequence) for seq_data in train_list])
    y_train = np.array([seq_data.microarray_signal for seq_data in train_list])

    # Prepare test data
    x_test = [seq_data.extracted_sequence(seq_length) for seq_data in test_list]
    #x_test = test_list
    # Create the model
    my_model = model((window_seq, 4), window, st, nt)

    # Fit the model on your training data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Prepare data for batch prediction
    all_encoded_windows = []
    sequence_index_map = []  # Maps each window to its original sequence index

    # Before starting the loop, ensure x_test is a list of SequenceData objects
    x_test = [seq_data.sequence for seq_data in test_list]  # This will extract the sequence strings

    # Then, when preparing data for batch prediction
    for sequence_index, sequence_str in enumerate(x_test):
        sequence_length = len(sequence_str)
        for i in range(0, sequence_length - window_seq + 1, 1):
            extracted_sequence = sequence_str[i:i + window_seq]
            encoded_sequence = one_hot_encoding(extracted_sequence)
            all_encoded_windows.append(encoded_sequence)
            sequence_index_map.append(sequence_index)

    # Perform batch predictions
    all_encoded_windows = np.array(all_encoded_windows)
    all_predictions = my_model.predict(all_encoded_windows).flatten()

    # Process predictions to find the max signal for each original sequence
    predictions = [[] for _ in range(len(x_test))]
    # Collecting all predictions
    for idx, prediction in zip(sequence_index_map, all_predictions):
        predictions[idx].append(prediction)  # Append the prediction to the list corresponding to idx
    # Save predictions to CSV
    sequences_test = [seq_data.sequence for seq_data in test_list]  # Extract full sequences for saving
    average_predictions = [np.mean(pred_list) if pred_list else float('nan') for pred_list in predictions]
    #integrated_gradients_example(my_model)

    df = pd.DataFrame({
        'Sequence': sequences_test,
        'Average Prediction': average_predictions
    })

    # Save DataFrame to CSV file
    csv_filename = 'microarray_files/central_imotif_sequences_varying_c.csv'  # Specify your desired path and file name

    append_predictions_to_csv(sequences_test, average_predictions, csv_filename)

    print("Predictions have been saved to the CSV file at:", csv_filename)







def main_graph():
    # Load the dataset
    file_train = 'microarray_files/final_table_microarray.csv'
    train_list = create_train_list(file_train)

    # Extract sequences and microarray signals
    sequences = [seq_data.sequence for seq_data in train_list]
    signals = [seq_data.microarray_signal for seq_data in train_list]
    ids = [seq_data.ID for seq_data in train_list]  # Assuming seq_data has an id attribute

    # Convert sequences to numerical features (one-hot encoding or other method)
    x = np.array([one_hot_encoding(seq) for seq in sequences])
    y = np.array(signals)

    # Split the data into training (90%) and testing (10%) sets
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(x, y, ids, test_size=0.1, random_state=42)

    with open('test_ids.txt', 'w') as file:
        for test_id in ids_test:
            file.write(f"{test_id}\n")

    # Train the model
    my_model = model((window_seq, 4), window, st, nt)
    my_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Predict on the testing set
    y_pred = my_model.predict(X_test)

    # Ensure y_pred is a flat array
    y_pred = y_pred.flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Calculate the Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(y_test, y_pred)
    print(f'Pearson Correlation: {correlation}, p-value: {p_value}')

    # Calculate the density of the points
    xy = np.vstack([y_test, y_pred])
    density = gaussian_kde(xy)(xy)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(y_test, y_pred, c=density, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density', fontsize=20)  # Increase font size for label
    cbar.ax.yaxis.set_tick_params(labelsize=20)  # Increase font size for ticks

    # Remove title and adjust fonts
    plt.xlabel('Measured propensities', fontsize=20)  # Increase font size here
    plt.ylabel('Predicted propensities', fontsize=20)  # Increase font size here
    plt.grid(True)

    # Add correlation coefficient, p-value, and number of points to the plot
    plt.annotate(f'R = {correlation:.2f}\np-value < 2.2e-16\nn = {len(y_test)}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=20, ha='left', va='top',  # Increase font size here
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white'))

    # Save the plot as an image file
    plt.savefig('performance_evaluation.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main_graph()

