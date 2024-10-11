
# iMotifPredictor

## Overview

iMotifPredictor is a convolutional neural network (CNN) designed to predict i-motif (iM) structures in DNA by integrating multiple data sources. iMs are non-canonical structures that form in single-stranded DNA and have been linked to various cellular functions and diseases. This project includes trained models for predicting iM structures using different data sources.

## Contents
- **AUROC**
Files used for calculating the AUROC (Area Under the Receiver Operating Characteristic curve) for each model with each type of negative dataset.

- **atac_files**
A zipped folder containing bedGraph files used to calculate ATAC signals and the corresponding signal files. Includes a Python script for calculating the ATAC signals.

- **fasta_files**
FASTA formatted sequences used in the analysis.

 - **genNullSeq**
Text files of the negative sequences generated using the GenNullSeq package.
Credit to the package:
```plaintext
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4937197/
```

- **im_seeker**
Results of running iM-Seeker and creating files for easier AUROC calculation.
Credit to:
```plaintext
https://academic.oup.com/nar/article/52/W1/W19/7659304
```


- **interpation_file**
A zipped file containing sequences created for model interpretation.

- **microarray_files**
Processed microarray data table used for training along with signal files for different sequences.
Credit:
```plaintext
https://academic.oup.com/nar/article/51/22/12020/7420102#429540511
```

- **pos_txt_files**
Processed positive sequences.
Credit:
```plaintext
https://academic.oup.com/nar/article/51/16/8309/7232843
```

- **random_neg**
Text files of randomly selected negative sequences.

- **txt_permutaion**
Text files of dishuffled negative sequences.

- **models**:
  - **atac_model_gen.h5**
  - **atac_model_rand.h5**
  - **atacandmicro_model_gen.h5**
  - **atacandmicro_model_rand.h5**
  - **iMPropensity.h5**
  - **micro_model_gen.h5**
  - **micro_model_perm.h5**
  - **micro_model_rand.h5**
  - **sequence_model_gen.h5**
  - **sequence_model_perm.h5**
  - **sequence_model_rand.h5**

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas

## Usage

The provided models can be used to predict iM structures based on different types of input data. Below are the instructions on how to use these models. Note that the provided code example is a suggested way to use the models, but you can also load sequences (in the appropriate format) and use the `predict` function to make predictions and save results as you prefer.

### Example Usage

Here is an example of how to load a model and make predictions using the provided models:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the function to load the model and make predictions
def predict_and_save(chunk_file, model_path, model_type):
    # Load the chunk data
    data = pd.read_csv(chunk_file)
    
    # Preprocess the sequences (placeholder, adjust as necessary)
    def encode_sequence(sequence):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence], dtype=np.int8)

    sequences = np.array([encode_sequence(seq) for seq in data['Sequence']])
    signal = np.array(data['signal']).reshape(-1, 1)
    atac_signal = np.array(data['atac_signal']).reshape(-1, 1)

    # Load the model
    model = load_model(model_path)

    # Make predictions based on the model type
    if model_type == 'sequence_model':
        predictions = model.predict(sequences, batch_size=128)
    elif model_type == 'micro_model':
        predictions = model.predict([sequences, signal], batch_size=128)
    elif model_type == 'atac_model':
        predictions = model.predict([sequences, atac_signal], batch_size=128)
    elif model_type == 'atacandmicro_model':
        predictions = model.predict([sequences, atac_signal, signal], batch_size=128)
    else:
        raise ValueError("Invalid model type")

    # Add the predictions to the dataframe
    prediction_column_name = model_path.split('/')[-1].replace('.h5', '')
    data[prediction_column_name] = predictions

    # Save the dataframe to the same file
    data.to_csv(chunk_file, index=False)

# Example usage
predict_and_save('input_chunk.csv', 'sequence_model_gen.h5', 'sequence_model')
```

### Input Format

- **DNA Sequences**: CSV file containing a column named `Sequence` with DNA sequences of the required length (either 60 or 124, depending on the model).
- **Microarray Signal**: Column named `signal` containing iM propensity scores predicted by the microarray model.
- **ATAC Signal**: Column named `atac_signal` containing open-chromatin signals.

### How to Use the Models

1. **Prepare Input Data**: Ensure your input data is in a CSV format with the necessary columns (`Sequence`, `signal`, and/or `atac_signal`).

2. **Load Model**: Use TensorFlow's `load_model` function to load the required model.

3. **Make Predictions**: Pass the input data to the model and obtain predictions.

4. **Save Predictions**: Add the predictions to your input data and save the results.

### Models Description

All models, except for iMPropensity, were trained on HEK293T data and predict the probability of iM formation:

- **sequence_model_gen.h5**: Model trained on GenNullSeq data , using sequences of length 124 encoded in one-hot format.
- **sequence_model_perm.h5**: Model trained on dishuffled data, using sequences of length 124 encoded in one-hot format.
- **sequence_model_rand.h5**: Model trained on randomly selected genomic sequences, using sequences of length 124 encoded in one-hot format.
- **micro_model_gen.h5**: Model trained on GenNullSeq data and iMpropensity signals, requires sequences of length 124 encoded in one-hot format and microarray signals.
- **micro_model_perm.h5**: Model trained on dishuffled data and iMpropensity signals, requires sequences of length 124 encoded in one-hot format and microarray signals.
- **micro_model_rand.h5**: Model trained on randomly selected data and iMpropensity signals, requires sequences of length 124 encoded in one-hot format and microarray signals.
- **atac_model_gen.h5**: Model trained on GenNullSeq genomic sequences and ATAC-seq data, requires sequences of length 124 encoded in one-hot format and ATAC signals.
- **atac_model_rand.h5**: Model trained on randomly selected sequences and ATAC-seq data, requires sequences of length 124 encoded in one-hot format and ATAC signals.
- **atacandmicro_model_gen.h5**: Model trained on GenNullSeq genomic sequences, iMpropensity data, and ATAC-seq data, requires sequences of length 124 encoded in one-hot format, microarray signals, and ATAC signals.
- **atacandmicro_model_rand.h5**: Model trained on randomly selected genomic, iMpropensity data , and ATAC-seq data, requires sequences of length 124 encoded in one-hot format, microarray signals, and ATAC signals.
- **iMPropensity.h5**: Model trained to predict iM propensity based on high-throughput microarray data, requires sequences of length 60 encoded in one-hot format.

### Using iMPropensity Model

The `iMPropensity.h5` model is used to predict the propensity of iM formation based on microarray data. This model receives a DNA sequence of length 60 encoded in one-hot format. Here's an example of how to use the `iMPropensity.h5` model:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define a function to encode sequences
def encode_sequence(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence], dtype=np.int8)

# Load the model
model = load_model('iMPropensity.h5')

# Example sequence
sequence = 'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT'
encoded_sequence = np.array([encode_sequence(sequence)])

# Make prediction
prediction = model.predict(encoded_sequence)
print(f'iM Propensity: {prediction[0][0]}')

# Example of sliding window approach
def sliding_window_prediction(sequence, model, window_size=60, stride=1):
    encoded_windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        window_seq = sequence[i:i+window_size]
        encoded_windows.append(encode_sequence(window_seq))
    encoded_windows = np.array(encoded_windows)
    predictions = model.predict(encoded_windows)
    return predictions

# Sliding window prediction for a sequence of length 124
sequence_124 = 'ACGT' * 31
predictions = sliding_window_prediction(sequence_124, model)
average_prediction = np.mean(predictions)
print(f'Average iM Propensity: {average_prediction}')
```

## Conclusion

iMotifPredictor provides a powerful tool for predicting i-motif structures in DNA using a combination of sequence information, microarray data, and open-chromatin information. By leveraging convolutional neural networks, iMotifPredictor achieves high accuracy and provides valuable insights into iM formation mechanisms.


