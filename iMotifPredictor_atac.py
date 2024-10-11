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
df_positive = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
df_negative = pd.read_csv('atac_files/HEK_iM_neg_SCORES', sep=',', header=0, names=columns)

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


# Define the SequenceData class
class SequenceData:
    def __init__(self, chromosome, sequence, classification, accessibility, start_coordinate, end_coordinate):
        self.chromosome = chromosome
        self.sequence = sequence
        self.classification = classification
        self.accessibility = accessibility
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate


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

    # Concatenate the output of the convolutional layer with the accessibility input
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

def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)
def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def createlistpos(positive_file,df_positive):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a list to store SequenceData objects for both positive and negative sequences
    train_data = []
    test_data = []

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    for match in matches:
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
            matching_rows = df_positive[
                (df_positive['Chromosome'] == chrom_parts[0]) & (df_positive['Start'] == start_pos) & (df_positive['End'] == end_pos)]
            accessibility_score = matching_rows['Score'].values[0]
            accessibility_score = math.log(accessibility_score + 1)
            if c_count_sequence >= c_count_complement:
                seq_data = SequenceData(chrom_parts[0], extracted_sequence, 1, accessibility_score, start_pos, end_pos)  # Assuming positive classification
                if re.search(r'\bchr1\b', chromosome):
                    test_data.append(seq_data)
                else:
                    train_data.append(seq_data)
            else:
                complement_seq_data = SequenceData(chrom_parts[0], complement, 1, accessibility_score, start_pos, end_pos)
                if re.search(r'\bchr1\b', chromosome):
                    test_data.append(complement_seq_data)
                else:
                    train_data.append(complement_seq_data)
    return test_data, train_data

def createlistposWD(positive_file,df_positive):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a list to store SequenceData objects for both positive and negative sequences
    list_data = []

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    for match in matches:
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
            matching_rows = df_positive[
                (df_positive['Chromosome'] == chrom_parts[0]) & (df_positive['Start'] == start_pos) & (df_positive['End'] == end_pos)]
            accessibility_score = matching_rows['Score'].values[0]
            accessibility_score = math.log(accessibility_score + 1)
            if c_count_sequence >= c_count_complement:
                seq_data = SequenceData(chrom_parts[0], extracted_sequence, 1, accessibility_score, start_pos, end_pos)  # Assuming positive classification
                list_data.append(seq_data)
            else:
                complement_seq_data = SequenceData(chrom_parts[0], complement, 1, accessibility_score, start_pos, end_pos)
                list_data.append(complement_seq_data)

    return list_data
def add_negatives_to_list(test_dict, train_dict, negative_file, df_negative):
    with open(negative_file, 'r') as file:
        data = file.read()
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_negative[
                (df_negative['Chromosome'] == chrom_parts[0]) & (df_negative['Start'] == start_pos) & (df_negative['End'] == end_pos)]
            if not matching_rows.empty:
                accessibility_score = matching_rows['Score'].values[0]
                accessibility_score = math.log(accessibility_score + 1)

            else:
                accessibility_score = None
            # Add the negative sequence to the appropriate dictionary with label 0
            if re.search(r'\bchr1\b', chromosome):
                test_dict.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos))
            else:
                train_dict.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos))

    return test_dict, train_dict
def add_negatives_to_listWD(listpos, negative_file, df_negative):
    with open(negative_file, 'r') as file:
        data = file.read()
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_negative[
                (df_negative['Chromosome'] == chrom_parts[0]) & (df_negative['Start'] == start_pos) & (df_negative['End'] == end_pos)]
            if not matching_rows.empty:
                accessibility_score = matching_rows['Score'].values[0]
                accessibility_score = math.log(accessibility_score + 1)

            else:
                accessibility_score = None
            # Add the negative sequence to the appropriate dictionary with label 0
            listpos.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos))


    return listpos

def main_random_access():
    positive_file= 'pos_txt_files/HEK_iM.txt'
    negative_file = 'random_neg/HEK_iM_neg.txt'
    df_positive = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative = pd.read_csv('atac_files/HEK_iM_neg_SCORES', sep=',', header=0, names=columns)


    # Create the lists to store SequenceData objects
    test_data, train_data = createlistpos(positive_file,df_positive)

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_list(test_data, train_data, negative_file,df_negative)

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
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)
    # First, get the predictions
    predictions = my_model.predict([x_test, x_test_accessibility])

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_random.csv'
    df.to_csv(csv_file_path, index=False)

    # Evaluate the model
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])


    return history

def main_random_access_wdlps():
    positive_file_train = 'pos_txt_files/HEK_iM.txt'
    negative_file_train = 'random_neg/HEK_iM_neg.txt'
    positive_file_test = 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test = 'random_neg/WDLPS_iM_neg.txt'
    df_positive_test = pd.read_csv('atac_files/WDLPS_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_test = pd.read_csv('atac_files/WDLPS_iM_neg_SCORES', sep=',', header=0, names=columns)
    df_positive_train = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_train = pd.read_csv('atac_files/HEK_iM_neg_SCORES', sep=',', header=0, names=columns)

    # Create the lists to store SequenceData objects
    test_data = createlistposWD(positive_file_test,df_positive_test)
    train_data= createlistposWD(positive_file_train,df_positive_train)

    # Add negative sequences to the appropriate lists
    test_data = add_negatives_to_listWD(test_data, negative_file_test,df_negative_test)
    train_data = add_negatives_to_listWD(train_data, negative_file_train,df_negative_train)
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
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)
    predictions = my_model.predict([x_test, x_test_accessibility])
    sequences_test = np.array([seq.sequence for seq in test_data])
    classifications_test = np.array([seq.classification for seq in test_data])
    predictions = np.array(predictions)  # Assuming predictions is already in a list format
    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_random_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)
    # Evaluate the model
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

def main_genNullSeq_access():
    positive_file= 'pos_txt_files/HEK_iM.txt'
    negative_file = 'genNellSeq/negHekiM.txt'
    df_positive = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative = pd.read_csv('atac_files/negHekiM_SCORES', sep=',', header=0, names=columns)
    # Create the lists to store SequenceData objects
    test_data, train_data = createlistpos(positive_file,df_positive)

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_list(test_data, train_data, negative_file,df_negative)

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
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)
    # First, get the predictions
    predictions = my_model.predict([x_test, x_test_accessibility])


    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_gen.csv'
    df.to_csv(csv_file_path, index=False)
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

def main_genNullSeq_access_wdlps():
    positive_file_train = 'pos_txt_files/HEK_iM.txt'
    negative_file_train = 'genNellSeq/negHekiM.txt'
    positive_file_test = 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test = 'genNellSeq/negWDLPSiM.txt'
    df_positive_test = pd.read_csv('atac_files/WDLPS_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_test = pd.read_csv('atac_files/negWDLPSiM_SCORES', sep=',', header=0, names=columns)
    df_positive_train = pd.read_csv('atac_files/HEK_iM_SCORES', sep=',', header=0, names=columns)
    df_negative_train = pd.read_csv('atac_files/negHekiM_SCORES', sep=',', header=0, names=columns)

    # Create the lists to store SequenceData objects
    test_data = createlistposWD(positive_file_test,df_positive_test)
    train_data= createlistposWD(positive_file_train,df_positive_train)

    # Add negative sequences to the appropriate lists
    test_data = add_negatives_to_listWD(test_data, negative_file_test,df_negative_test)
    train_data = add_negatives_to_listWD(train_data, negative_file_train,df_negative_train)

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
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)
    predictions = my_model.predict([x_test, x_test_accessibility])
    sequences_test = np.array([seq.sequence for seq in test_data])
    classifications_test = np.array([seq.classification for seq in test_data])
    predictions = np.array(predictions)  # Assuming predictions is already in a list format
    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_acc_gen_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)

    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

history_random = main_random_access_wdlps()
history_genNull = main_genNullSeq_access_wdlps()