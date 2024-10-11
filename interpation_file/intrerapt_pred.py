import random
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import re
import pandas as pd




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

    def __init__(self, sequence, coordinate, label_nuc, label=None):
        self.sequence = sequence
        self.coordinate = coordinate
        self.label = label
        self.label_middle = label_nuc
        self.prediction = None

    def set_prediction(self, prediction):
        self.prediction = prediction

    def get_encoded_sequence(self):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.33, 0, 0.33, 0.33]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in self.sequence], dtype=np.int8)


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

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=in1, outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

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

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = extracted_sequence.count('C')
            g_count_complement = complement.count('C')

            sequence_to_use = extracted_sequence if g_count_sequence >= g_count_complement else complement
            train_sequences.append(Sequence(sequence_to_use, chromosome, 0, 1))

    return train_sequences


def add_negatives_to_list(sequence_list, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            sequence_list.append(Sequence(extracted_sequence, chromosome, 0, 0))

    return sequence_list


# Main function
if __name__ == "__main__":
    positive_file = '../pos_txt_files/HEK_iM.txt'
    negative_file_train = '../random_neg/HEK_iM_neg.txt'

    # Generate training sequences
    train_sequences = createTrainlist()
    train_sequences = add_negatives_to_list(train_sequences, negative_file_train)
    random.shuffle(train_sequences)
    # Calculate the size of the validation set (10% of the training data)
    validation_size = int(len(train_sequences) * 0.1)

    # Split the training data into training and validation sets
    validation = train_sequences[:validation_size]
    training = train_sequences[validation_size:]

    # Prepare training data
    x_train = np.array([seq.get_encoded_sequence() for seq in training])
    y_train = np.array([seq.label for seq in training])
    x_valid = np.array([seq.get_encoded_sequence() for seq in validation])
    y_valid = np.array([seq.label for seq in validation])

    # Create and fit the model
    nn_model = model(x_train.shape[1:], window, st, nt)
    nn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
    test_scores = nn_model.evaluate(x_valid, y_valid)
    # Load the CSV file containing the sequences
    sequences_df = pd.read_csv(
        'central_imotif_sequences_varying_c.csv')  # Replace with the actual path to your CSV file

    # Convert strings to Sequence objects and get encoded sequences
    sequence_objects = [Sequence(seq, '', '', '') for seq in sequences_df['Sequence']]
    encoded_sequences = np.array([seq_obj.get_encoded_sequence() for seq_obj in sequence_objects])

    # Predict using the model
    predictions = nn_model.predict(encoded_sequences)

    # Add the predictions as a new column to the dataframe
    sequences_df['Prediction_rand'] = predictions.flatten()

    # Save the dataframe with the new predictions to the same CSV file
    sequences_df.to_csv('central_imotif_sequences_varying_c.csv',
                        index=False)  # Replace with your desired file path

    print("Predictions have been added to the CSV file.")
