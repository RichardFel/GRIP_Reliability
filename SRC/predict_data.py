import numpy as np
from SRC.utils import *
import pandas as pd
import numpy as np
import tensorflow as tf

def mean_value(chunk):
    return np.round(np.mean(chunk) + 0.01)

def predict_data(data_df, settings, file_name, model):
    data_df = data_df.reset_index()
    data_df['group'] = np.nan
    samples_per_tensec = int(settings['sample_size'] * settings['frequency'])
    samples = int(np.floor(len(data_df) / samples_per_tensec))
    
    for num in range(samples):
        start = samples_per_tensec * num
        end = samples_per_tensec * num + samples_per_tensec - 1
        data_df.loc[start:end, 'group'] = int(num)
    data_df = data_df.dropna()
        
    # Drop values that are
    data_input_array = {part: group[['acc_x', 'acc_y', 'acc_z']]
                        for part, group in data_df.groupby('group')}
    length, width = data_input_array[list(
        data_input_array.keys())[0]].shape
    data_np = np.empty((len(data_input_array), length, width))
    counter = 0
    for _, item in data_input_array.items():
        data_np[counter] = item.values
        counter += 1

    # Load model
    model = tf.keras.models.load_model(f'{settings["MODELS_DIR"]}/{model}')

    # predict values
    # Drop
    predicted_values = model.predict(data_np)
    
    predicted_values = np.argmax(predicted_values, axis=1)
    predicted_values_df = pd.DataFrame(
        predicted_values, columns=['activities [10 sec]'])
    predicted_values_df = predicted_values_df.merge(data_df.drop_duplicates(subset='group')[['time_2hours', 'group','time']],
                                                    left_index=True,  # Use index from predicted_values_df
                                                    right_on='group').reset_index(drop=True)
    predicted_values_df.to_csv(f'data/ten_sec/{file_name}_10sec.csv')

    # values per minute
    # Define the chunk size
    chunks = [predicted_values[i:i + settings['chunk_size']]
              for i in range(0, len(predicted_values), settings['chunk_size'])]

    most_common_per_chunk = [mean_value(chunk) for chunk in chunks[:-1]]
    most_common_per_chunk_df = pd.DataFrame(
        most_common_per_chunk, columns=['activities [1min]'])
    most_common_per_chunk_df = most_common_per_chunk_df.merge(predicted_values_df[::6].reset_index()[['time_2hours','time']],
                                                              left_index=True,  # Merge on the index of most_common_per_chunk
                                                              right_index=True)
    most_common_per_chunk_df.to_csv(f'data/one_min/{file_name}_1min.csv')

    return most_common_per_chunk_df
