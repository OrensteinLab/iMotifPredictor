import random
import re
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import logomaker
import pandas as pd
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
def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)
def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def createDict():
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a dictionary to store both positive and negative sequences with classifications
    train_dict = {}
    test_dict = {}

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)
    stop_index = int(len(matches) * 1)

    for match in matches[:stop_index]:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = extracted_sequence.count('C')
            c_count_complement = complement.count('C')

            if c_count_sequence >= c_count_complement:
                if  re.search(r'\bchr1\b', chromosome):
                    test_dict[extracted_sequence] = 1
                else:
                    train_dict[extracted_sequence] = 1
            else:
                if  re.search(r'\bchr1\b', chromosome):
                    test_dict[complement] = 1
                else:
                    train_dict[complement] = 1
    return test_dict, train_dict

def createDictWDLPS():
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a dictionary to store both positive and negative sequences with classifications
    train_dict = {}
    test_dict = {}

    with open(positive_file_test, 'r') as file:
        data_test = file.read()
    with open(positive_file_train, 'r') as file:
        data_train = file.read()
    # Use re.findall to extract all positive sequences
    matches_test = re.findall(pattern, data_test)
    matches_train = re.findall(pattern, data_train)
    for match in matches_test:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = sequence.count('C')
            c_count_complement = complement.count('C')

            if c_count_sequence >= c_count_complement:
                test_dict[extracted_sequence] = 1
            else:
                test_dict[complement] = 1
    for match in matches_train:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            c_count_sequence = sequence.count('C')
            c_count_complement = complement.count('C')

            if c_count_sequence >= c_count_complement:
                   train_dict[extracted_sequence] = 1
            else:
                  train_dict[complement] = 1
    return test_dict, train_dict
def add_negatives_to_dict(test_dict, train_dict,negative_file):
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
        if (len(extracted_sequence) == 124):
            # Add the negative sequence to the appropriate dictionary with label 0
            if re.search(r'\bchr1\b', chromosome):
                test_dict[extracted_sequence] = 0
            else:
                train_dict[extracted_sequence] = 0

    return test_dict, train_dict





def add_negatives_to_dict_gen(test_dict, train_dict, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    # Define a regular expression pattern to match sequences with headers
    pattern = r'>([^\n]+)\n([ACGTacgt\n]+)'
    matches = re.findall(pattern, data)

    for header, sequence in matches:
        sequence = sequence.replace('\n', '')  # Remove newline characters within the sequence
        sequence = sequence.upper()

        # Check if the header contains chr1_
        if re.search(r'^chr1_', header):
            test_dict[sequence] = 0
        else:
            train_dict[sequence] = 0

    return test_dict, train_dict



def add_negatives_to_dict_WDLPS(dic, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    # Assuming the negative sequences are stored line by line in the file
    neg_sequences = data.strip().split('\n')
    for sequence in neg_sequences:
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            sequence = sequence.upper()
            dic[extracted_sequence] = 0

    return dic
def plot_metrics(history1, history2, history3, title1, title2, title3):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 9))

    # Plot AUC values
    plt.subplot(3, 1, 3)
    plt.plot(history1.history['auc'], label=f'{title1} - Train AUC')
    plt.plot(history1.history['val_auc'], label=f'{title1} - Validation AUC')
    plt.plot(history2.history['auc_1'], label=f'{title2} - Train AUC', linestyle='dashed')
    plt.plot(history2.history['val_auc_1'], label=f'{title2} - Validation AUC', linestyle='dashed')
    plt.plot(history3.history['auc_2'], label=f'{title3} - Train AUC', linestyle='dotted')
    plt.plot(history3.history['val_auc_2'], label=f'{title3} - Validation AUC', linestyle='dotted')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.suptitle(f"{title1} vs {title2} vs {title3}")
    plt.tight_layout()
    plt.show()
# Integrated Gradients functions
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

    # Calculate integrated gradients
    attributions = integrated_gradients(one_hot_sequence, my_model, target_class_idx)

    # Visualize the integrated gradients
    plot_integrated_gradients_logomaker(attributions, sequence)

#####################################################################
positive_file= 'pos_txt_files/HEK_iM.txt'
positive_file_train= 'pos_txt_files/HEK_iM.txt'
positive_file_test= 'pos_txt_files/WDLPS_iM.txt'

#main with permutaions
def main_permutions():
    negative_file = 'txt_permutaion/HEK_iM_perm_neg.txt'
    negative_file_train = 'txt_permutaion/HEK_iM_perm_neg.txt'
    negative_file_test = 'txt_permutaion/WDLPS_iM_perm_neg.txt'
    test_dict,train_dict = createDictWDLPS()
    #test_dict, train_dict = createDict()
    #test_dict, train_dict = add_negatives_to_dict(test_dict,train_dict,negative_file)
    test_dict = add_negatives_to_dict_WDLPS(test_dict, negative_file_test)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    # Sizes of train and test dictionaries
    print("Train set size:", len(train_dict))  # Number of samples in the train set
    print("Test set size:", len(test_dict))  # Number of samples in the test set

    items_train = list(train_dict.items())
    random.shuffle(items_train)
    shuffled_dict_train = dict(items_train)
    items_test = list(test_dict.items())
    random.shuffle(items_test)
    shuffled_dict_test = dict(items_test)
    # Extract sequences and classifications from the dictionary
    sequences_train = list(shuffled_dict_train.keys())
    classifications_train = [shuffled_dict_train[seq] for seq in shuffled_dict_train]
    sequences_test = list(shuffled_dict_test.keys())
    classifications_test = [shuffled_dict_test[seq] for seq in shuffled_dict_test]
    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    # Define your target labels (y) based on the classifications
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    # Integrated gradients visualization for the test sequence
    integrated_gradients_example(my_model)

    # Evaluate the model
    test_scores = my_model.evaluate(x_test, y_test)
    # First, get the predictions
    predictions = my_model.predict(x_test)

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_seq_perm_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])
    #plot_metric(history, title="Permutations Method")
    return history


#main with rangom genertive
def main_random():
    negative_file= 'random_neg/HEK_iM_neg.txt'
    positive_file_train= 'pos_txt_files/HEK_iM.txt'
    negative_file_train= 'random_neg/HEK_iM_neg.txt'
    positive_file_test= 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test= 'random_neg/WDLPS_iM_neg.txt'
    test_dict, train_dict = createDictWDLPS()
    #test_dict, train_dict = createDict()
    #test_dict, train_dict = add_negatives_to_dict(test_dict,train_dict,negative_file)
    test_dict = add_negatives_to_dict_WDLPS(test_dict, negative_file_test)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    items_train = list(train_dict.items())
    random.shuffle(items_train)
    shuffled_dict_train = dict(items_train)
    items_test = list(test_dict.items())
    random.shuffle(items_test)
    shuffled_dict_test = dict(items_test)
    # Extract sequences and classifications from the dictionary
    sequences_train = list(shuffled_dict_train.keys())
    classifications_train = [shuffled_dict_train[seq] for seq in shuffled_dict_train]
    sequences_test = list(shuffled_dict_test.keys())
    classifications_test = [shuffled_dict_test[seq] for seq in shuffled_dict_test]
    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    # Define your target labels (y) based on the classifications
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Compile the model
    # my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    # Fit the model on your data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    integrated_gradients_example(my_model)

    # Evaluate the model
    test_scores = my_model.evaluate(x_test, y_test)
    predictions = my_model.predict(x_test)

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_seq_random_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"Predictions and true labels have been saved to {csv_file_path}.")
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])
    #plot_metric(history, title="Random Method")
    return history

def main_genNullSeq():
    negative_file= 'genNellSeq/negHekiM.txt'
    positive_file_train= 'pos_txt_files/HEK_iM.txt'
    negative_file_train= 'genNellSeq/negHekiM.txt'
    positive_file_test= 'pos_txt_files/WDLPS_iM.txt'
    negative_file_test= 'genNellSeq/negWDLPSiM.txt'
    #test_dict, train_dict = createDict()
    test_dict,train_dict = createDictWDLPS()
    test_dict = add_negatives_to_dict_WDLPS(test_dict, negative_file_test)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    #test_dict,train_dict= add_negatives_to_dict(test_dict,train_dict,negative_file)
    items_train = list(train_dict.items())
    random.shuffle(items_train)
    shuffled_dict_train = dict(items_train)
    items_test = list(test_dict.items())
    random.shuffle(items_test)
    shuffled_dict_test = dict(items_test)
    # Extract sequences and classifications from the dictionary
    sequences_train = list(shuffled_dict_train.keys())
    classifications_train = [shuffled_dict_train[seq] for seq in shuffled_dict_train]
    sequences_test = list(shuffled_dict_test.keys())
    classifications_test = [shuffled_dict_test[seq] for seq in shuffled_dict_test]
    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    # Define your target labels (y) based on the classifications
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    integrated_gradients_example(my_model)

    # Evaluate the model
    test_scores = my_model.evaluate(x_test, y_test)
    # First, get the predictions
    predictions = my_model.predict(x_test)

    # Then, you can create a DataFrame and save it as before
    import pandas as pd

    # Assuming y_test are your true labels
    df = pd.DataFrame({
        'True_Labels': y_test.flatten(),  # Adjust this if your labels are not already in a 1D format
        'Predictions': predictions.flatten()  # Adjust if predictions are not in the format you expect
    })

    # Save the DataFrame to a CSV file
    csv_file_path = 'AUROC/predictions_and_true_labels_seq_gen_WDLPS.csv'
    df.to_csv(csv_file_path, index=False)
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

history_permutations = main_permutions()
history_random = main_random()
history_genNull = main_genNullSeq()

#plot_metrics(history_permutations, history_random,history_genNull, "Permutations Method", "Random Method","genNullSeq Method")