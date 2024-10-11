import random
import pandas as pd

import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import re
import numpy as np
import math

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
seq_lengh = 124
# Reading the data from the file into a DataFrame
columns = ['Chromosome', 'Start', 'End', 'Score']
df_microarray = pd.read_csv('microarray_files/signals_data_HEK_iM.csv')

import pandas as pd


def save_sequence_data_to_csv(sequences, classifications, predictions, chromatin_accessibility, filename):
    data = {
        'Sequence': sequences,
        'Classification': classifications,
        'Prediction': [prediction[0] for prediction in predictions],
        'Chromatin_Accessibility': chromatin_accessibility
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
def match_microarray_signal(sequence, df_microarray):
    # Example function to find the corresponding microarray signal for a given sequence
    # This is just a placeholder. You'll need to iMplement the actual matching logic.
    matched_row = df_microarray[df_microarray['Sequence'] == sequence]
    if not matched_row.empty:
        return matched_row['Signal'].iloc[0]
    else:
        return None  # or some default value


# Define the SequenceData class
class SequenceData:
    def __init__(self, chromosome, sequence, classification, accessibility, start_coordinate, end_coordinate,micro):
        self.chromosome = chromosome
        self.sequence = sequence
        self.classification = classification
        self.accessibility = accessibility
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate
        self.microarray_signal=micro

def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)
    in2 = Input(shape=(1,))  # Additional input for accessibility scores
    in3 = Input(shape=(1,)) # Microarray signal

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)

    # Concatenate the output of the convolutional layer with the accessibility input
    merged = tensorflow.keras.layers.concatenate([pool, in2, in3])

    # Creating Hidden Layer
    hidden1 = Dense(fc)(merged)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=[in1, in2, in3], outputs=output)

    opt = tensorflow.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tensorflow.keras.metrics.AUC()])

    return mdl





def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)


def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)


def createlistpos(positive_file, df_positive, df_microarray):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create lists to store SequenceData objects for both positive and negative sequences
    train_data = []
    test_data = []

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    # Extract the microarray signals
    microarray_signals = df_microarray['Average Prediction'].tolist()

    for i, match in enumerate(matches):
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = extracted_sequence.count('C')
            c_count_complement = complement.count('C')
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_positive[
                (df_positive['Chromosome'] == chrom_parts[0]) & (df_positive['Start'] == start_pos) & (
                            df_positive['End'] == end_pos)]
            accessibility_score = matching_rows['Score'].values[0] if not matching_rows.empty else None
            accessibility_score = math.log(accessibility_score + 1) if accessibility_score is not None else None

            microarray_signal = microarray_signals[i] if i < len(microarray_signals) else None

            if c_count_sequence >= c_count_complement:
                seq_data = SequenceData(chrom_parts[0], extracted_sequence, 1, accessibility_score, start_pos, end_pos, microarray_signal)
            else:
                seq_data = SequenceData(chrom_parts[0], complement, 1, accessibility_score, start_pos, end_pos, microarray_signal)

            if re.search(r'\bchr1\b', chromosome):
                test_data.append(seq_data)
            else:
                train_data.append(seq_data)

    return test_data, train_data


def createlistposWD(positive_file, df_positive, df_microarray):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a list to store SequenceData objects for both positive and negative sequences
    list_data = []

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)
    microarray_signals = df_microarray['Average Prediction'].tolist()
    for i, match in enumerate(matches):
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = extracted_sequence.count('C')
            c_count_complement = complement.count('C')
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            microarray_signal = microarray_signals[i] if i < len(microarray_signals) else None
            matching_rows = df_positive[
                (df_positive['Chromosome'] == chrom_parts[0]) & (df_positive['Start'] == start_pos) & (
                            df_positive['End'] == end_pos)]
            accessibility_score = matching_rows['Score'].values[0]
            accessibility_score = math.log(accessibility_score + 1)
            if c_count_sequence >= c_count_complement:
                seq_data = SequenceData(chrom_parts[0], extracted_sequence, 1, accessibility_score, start_pos,
                                        end_pos,microarray_signal)  # Assuming positive classification
                list_data.append(seq_data)
            else:
                complement_seq_data = SequenceData(chrom_parts[0], complement, 1, accessibility_score, start_pos,
                                                   end_pos,microarray_signal)
                list_data.append(complement_seq_data)

    return list_data

def add_negatives_to_list(test_dict, train_dict, negative_file, df_negative, df_microarray_negative):
    with open(negative_file, 'r') as file:
        data = file.read()
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)

    # Extract the microarray signals
    microarray_signals = df_microarray_negative['Average Prediction'].tolist()

    for i, match in enumerate(matches):
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_negative[
                (df_negative['Chromosome'] == chrom_parts[0]) & (df_negative['Start'] == start_pos) & (
                            df_negative['End'] == end_pos)]
            accessibility_score = None
            if not matching_rows.empty:
                accessibility_score = math.log(matching_rows['Score'].values[0] + 1)

            # Directly use the microarray signal based on the index
            microarray_signal = microarray_signals[i] if i < len(microarray_signals) else None

            seq_data = SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos, microarray_signal)
            if re.search(r'\bchr1\b', chromosome):
                test_dict.append(seq_data)
            else:
                train_dict.append(seq_data)

    return test_dict, train_dict




def add_negatives_to_listWD(listpos, negative_file, df_negative,df_microarray_negative):
    with open(negative_file, 'r') as file:
        data = file.read()
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    microarray_signals = df_microarray_negative['Average Prediction'].tolist()
    for i, match in enumerate(matches):
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            microarray_signal = microarray_signals[i] if i < len(microarray_signals) else None

            matching_rows = df_negative[
                (df_negative['Chromosome'] == chrom_parts[0]) & (df_negative['Start'] == start_pos) & (
                            df_negative['End'] == end_pos)]
            if not matching_rows.empty:
                accessibility_score = matching_rows['Score'].values[0]
                accessibility_score = math.log(accessibility_score + 1)

            else:
                accessibility_score = None
            # Add the negative sequence to the appropriate dictionary with label 0
            listpos.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos,microarray_signal))

    return listpos


def main_random_access():
    positive_file = 'pos_txt_files\HEK_iM.txt'
    negative_file = 'random_neg\HEK_iM_neg.txt'
    df_positive = pd.read_csv('atac_files\HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative = pd.read_csv('atac_files\HEK_iM_neg_SCORES', sep=',', header=0, names=columns)
    df_microarray_negative = pd.read_csv('microarray_files\signals_data_HEK_iM_neg.csv')
    # Create the lists to store SequenceData objects
    test_data, train_data = createlistpos(positive_file, df_positive,df_microarray)

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_list(test_data, train_data, negative_file, df_negative,df_microarray_negative)

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences and classifications from the list of SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)

    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)
    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)
    # Extract microarray signals from SequenceData objects
    # Convert microarray_signal to float, handling any missing values
    x_train_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in train_data]).reshape(-1, 1)
    # Convert microarray_signal to float, handling any missing values
    x_test_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in test_data]).reshape(-1, 1)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)



    # Convert arrays to a consistent dtype if needed
    x_train_microarray = x_train_microarray.astype(np.float32)

    # Fit the model (try with individual arrays if necessary to isolate the issue)
    history = my_model.fit(
        [x_train, x_train_accessibility, x_train_microarray],
        y_train,
        batch_size=batch_size,
        epochs=epochs
        )

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility, x_test_microarray], y_test)
    # First, get the predictions
    predictions = my_model.predict([x_test, x_test_accessibility, x_test_microarray])

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten(),# Adjust if predictions are not in the format you expect
        'microarray_signal': x_test_microarray.flatten(),
        'atac_accessibility': x_test_accessibility.flatten()
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_mic_ran.csv'
    df.to_csv(csv_file_path, index=False)
    # Evaluate the model
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history


def main_random_access_WD():
    positive_file_train = 'pos_txt_files/HEK_iM.txt'
    negative_file_train = 'random_neg/HEK_iM_neg.txt'
    positive_file_test = 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test = 'random_neg/WDLPS_iM_neg.txt'
    df_microarray_train = pd.read_csv('microarray_files/signals_data_HEK_iM.csv')
    df_microarray_train_neg = pd.read_csv('microarray_files/signals_data_HEK_iM_neg.csv')
    df_microarray_test = pd.read_csv('microarray_files/signals_data_WDLPS_iM.csv')
    df_microarray_test_neg = pd.read_csv('microarray_files/signals_data_WDLPS_iM_neg.csv')
    df_positive_test = pd.read_csv('atac_files/WDLPS_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_test = pd.read_csv('atac_files/WDLPS_iM_neg_SCORES', sep=',', header=0, names=columns)
    df_positive_train = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_train = pd.read_csv('atac_files/HEK_iM_neg_SCORES', sep=',', header=0, names=columns)

    # Create the lists to store SequenceData objects
    test_data = createlistposWD(positive_file_test, df_positive_test, df_microarray_test)
    train_data = createlistposWD(positive_file_train, df_positive_train, df_microarray_train)

    # Add negative sequences to the appropriate lists
    test_data = add_negatives_to_listWD(test_data, negative_file_test, df_negative_test, df_microarray_test_neg)
    train_data = add_negatives_to_listWD(train_data, negative_file_train, df_negative_train, df_microarray_train_neg)

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences, classifications, and microarray signals from SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)
    x_train_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in train_data]).reshape(-1, 1)
    x_test_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in test_data]).reshape(-1, 1)


    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility, classifications, and microarray signals)
    history = my_model.fit([x_train, x_train_accessibility, x_train_microarray], y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility, x_test_microarray], y_test)

    # First, get the predictions
    predictions = my_model.predict([x_test, x_test_accessibility, x_test_microarray])

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten(),
        'microarray_signal': x_test_microarray.flatten(),
        'atac_accessibility': x_test_accessibility.flatten()
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_mic_ran_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)

    # Save results and print evaluation metrics
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history



def main_genNullSeq_access_WDLPS():
    positive_file_train = 'pos_txt_files/HEK_iM.txt'
    negative_file_train = 'genNellSeq/negHekiM.txt'
    positive_file_test = 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test = 'genNellSeq/negWDLPSiM.txt'
    df_microarray_train = pd.read_csv('microarray_files/signals_data_HEK_iM.csv')
    df_microarray_train_neg = pd.read_csv('microarray_files/signals_data_negHekiM.csv')
    df_microarray_test = pd.read_csv('microarray_files/signals_data_WDLPS_iM.csv')
    df_microarray_test_neg = pd.read_csv('microarray_files/signals_data_negWDLPSiM.csv')
    df_positive_test = pd.read_csv('atac_files/WDLPS_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_test = pd.read_csv('atac_files/negWDLPSiM_SCORES', sep=',', header=0, names=columns)
    df_positive_train = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_train = pd.read_csv('atac_files/negHekiM_SCORES', sep=',', header=0, names=columns)

    # Create the lists to store SequenceData objects
    test_data = createlistposWD(positive_file_test, df_positive_test, df_microarray_test)
    train_data = createlistposWD(positive_file_train, df_positive_train, df_microarray_train)

    # Add negative sequences to the appropriate lists
    test_data = add_negatives_to_listWD(test_data, negative_file_test, df_negative_test, df_microarray_test_neg)
    train_data = add_negatives_to_listWD(train_data, negative_file_train, df_negative_train, df_microarray_train_neg)

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences, classifications, and microarray signals from SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)
    x_train_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                   for seq in train_data]).reshape(-1, 1)
    x_test_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in test_data]).reshape(-1, 1)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility, classifications, and microarray signals)
    history = my_model.fit([x_train, x_train_accessibility, x_train_microarray], y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility, x_test_microarray], y_test)

    # First, get the predictions
    predictions = my_model.predict([x_test, x_test_accessibility, x_test_microarray])

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten(),  # Adjust if predictions are not in the format you expect
        'microarray_signal': x_test_microarray.flatten(),
        'atac_accessibility': x_test_accessibility.flatten()
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_mic_gen_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)

    # Save results and print evaluation metrics
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history




def main_genNullSeq():
    positive_file = 'pos_txt_files/HEK_iM.txt'
    negative_file = 'genNellSeq/negHEKiM.txt'
    df_positive = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative = pd.read_csv('atac_files/negHEKiM_SCORES', sep=',', header=0, names=columns)
    df_microarray_negative = pd.read_csv('microarray_files/signals_data_negHekiM.csv')

    # Create the lists to store SequenceData objects
    test_data, train_data = createlistpos(positive_file, df_positive, df_microarray)

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_list(test_data, train_data, negative_file, df_negative, df_microarray_negative)


    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences and classifications from the list of SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)

    # Extract microarray signals from SequenceData objects
    # Convert microarray_signal to float, handling any missing values
    x_train_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in train_data]).reshape(-1, 1)
    x_test_microarray = np.array([float(seq.microarray_signal) if seq.microarray_signal is not None else 0.0
                                  for seq in test_data]).reshape(-1, 1)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit(
        [x_train, x_train_accessibility, x_train_microarray],
        y_train,
        batch_size=batch_size,
        epochs=epochs
    )
    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility, x_test_microarray], y_test)
    predictions = my_model.predict([x_test, x_test_accessibility, x_test_microarray])

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten() , # Adjust if predictions are not in the format you expect
        'microarray_signal': x_test_microarray.flatten(),
        'atac_accessibility': x_test_accessibility.flatten()
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_mic_gen.csv'
    df.to_csv(csv_file_path, index=False)
    # Save results and print evaluation metrics
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

main_random_access_WD()
main_genNullSeq_access_WDLPS()
#main_genNullSeq()
#main_random_access()
