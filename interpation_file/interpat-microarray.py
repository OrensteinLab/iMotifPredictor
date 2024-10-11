import time
import random
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import pandas as pd
from Bio import SeqIO
import sys
import re

# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 5
window = random.randint(5, 20)
st = random.randint(1, 10)
nt = random.randint(1, 10)
seq_length = 124



class Sequence:

    def __init__(self, sequence, coordinate, label_nuc, label=None, signal=0):
        self.sequence = sequence
        self.coordinate = coordinate
        self.label = label
        self.label_middle = label_nuc
        self.prediction = None
        self.signal = signal

    def set_prediction(self, prediction):
        self.prediction = prediction

    def get_encoded_sequence(self):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.33, 0, 0.33, 0.33]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in self.sequence], dtype=np.int8)


def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)
    in2 = Input(shape=(1,))  # Additional input for accessibility scores

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)
    merged = tensorflow.keras.layers.concatenate([pool, in2])

    # Creating Hidden Layer
    hidden1 = Dense(fc)(merged)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=[in1, in2], outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = tensorflow.keras.optimizers.legacy.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tensorflow.keras.metrics.AUC()])

    return mdl


def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence}
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)


def createTrainlist():
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    train_sequences = []

    with open(positive_file, 'r') as file:
        data = file.read()

    matches = re.findall(pattern, data)
    # Extract the microarray signals
    microarray_signals = df_microarray['Signal'].tolist()

    for i, match in enumerate(matches):

        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = extracted_sequence.count('C')
            g_count_complement = complement.count('C')
            microarray_signal = microarray_signals[i] if i < len(microarray_signals) else None

            sequence_to_use = extracted_sequence if g_count_sequence >= g_count_complement else complement
            train_sequences.append(Sequence(sequence_to_use, chromosome, 0, 1, microarray_signal))

    return train_sequences


def add_negatives_to_list(sequence_list, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    # Extract the microarray signals
    microarray_signals = df_microarray_negative['Signal'].tolist()

    for i, match in enumerate(matches):
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        # Directly use the microarray signal based on the index
        microarray_signal = microarray_signals[i] if i < len(microarray_signals) else 0

        if len(extracted_sequence) == 124:
            sequence_list.append(Sequence(extracted_sequence, chromosome, 0, 0, microarray_signal))

    return sequence_list

def attach_signals_to_sequences(sequence_list, signal_data):
    """
    Attaches signals to sequences based on some matching criteria.

    :param sequence_list: List of Sequence objects.
    :param signal_data: DataFrame with signal data; must have 'Sequence' and 'Signal' columns.
    :return: Updated list of Sequence objects with signals attached.
    """
    # Convert DataFrame to dictionary for easier lookup
    signal_dict = pd.Series(signal_data.Signal.values, index=signal_data.Sequence).to_dict()

    for seq_obj in sequence_list:
        # Attempt to fetch the signal for the current sequence; default to 0 if not found
        seq_obj.signal = signal_dict.get(seq_obj.sequence, 0)

    return sequence_list
# Main function
if __name__ == "__main__":

    # Part 1: Training the model with training data
    positive_file = '../pos_txt_files/HEK_iM.txt'
    negative_file_train = '../txt_permutaion/HEK_iM_perm_neg.txt'
    # Generate training sequences
    df_microarray = pd.read_csv('../microarray_files/signals_data_HEK_iM.csv')
    df_microarray_negative = pd.read_csv('../microarray_files/signals_data_HEK_iM_perm.csv')
    train_sequences = createTrainlist()
    train_sequences = add_negatives_to_list(train_sequences, negative_file_train)
    random.shuffle(train_sequences)

    # Prepare training data
    x_train = np.array([seq.get_encoded_sequence() for seq in train_sequences])
    x_train_signal = np.array(
        [float(seq.signal) if seq.signal is not None else 0.0 for seq in train_sequences]).reshape(-1, 1)

    y_train = np.array([seq.label for seq in train_sequences])
    # Calculate the size of the validation set (10% of the training data)
    validation_size = int(len(train_sequences) * 0.1)

    # Split the training data into training and validation sets
    validation = train_sequences[:validation_size]
    training = train_sequences[validation_size:]
    # Prepare training data
    x_train = np.array([seq.get_encoded_sequence() for seq in training])
    x_train_signal = np.array([seq.signal if seq.signal is not None else 0.0 for seq in training]).reshape(-1, 1)

    y_train = np.array([seq.label for seq in training])
    x_valid = np.array([seq.get_encoded_sequence() for seq in validation])
    x_valid_signal = np.array([seq.signal if seq.signal is not None else 0.0 for seq in validation]).reshape(-1, 1)

    y_valid = np.array([seq.label for seq in validation])
    # Create and fit the model
    nn_model = model(x_train.shape[1:], window, st, nt)
    nn_model.fit([x_train, x_train_signal], y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=([x_valid, x_valid_signal], y_valid))
    test_scores = nn_model.evaluate([x_valid, x_valid_signal], y_valid)
    sequences_df = pd.read_csv('mutations.csv')
    # Extract sequences and signals for model input
    sequences = sequences_df['Mutated_Sequence'].tolist()
    signals = sequences_df['Signal'].values.reshape(-1, 1)  # Assuming 'Signal' column exists and is relevant
    sequence_objects = [Sequence(seq, '', '', '') for seq in sequences]
    #train_sequences_with_signals = attach_signals_to_sequences(sequence_objects, signals_df)
    encoded_sequences = np.array([seq_obj.get_encoded_sequence() for seq_obj in sequence_objects])
    #encoded_signals = np.array([seq_obj.signal for seq_obj in sequence_objects]).reshape(-1, 1)
    # Predict using the model
    predictions = nn_model.predict([encoded_sequences, signals])
    # Add the predictions as a new column to the dataframe
    sequences_df['Prediction_perm_mic'] = predictions.flatten()
    # Save the dataframe with the new predictions to the same CSV file
    sequences_df.to_csv('mutations.csv',
                        index=False)  # Replace with your desired file path
    print("Predictions have been added to the CSV file.")




