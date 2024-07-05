#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
import librosa
from scipy.stats import skew, kurtosis, iqr, mode
from sklearn.metrics import accuracy_score
from librosa import load, resample, feature
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split,KFold
import os
import pandas as pd







warnings.filterwarnings("ignore")









directory = '/Users/aman/Downloads/SpeechCommand'
files = os.listdir(directory)
classes_list=[]
audio_classes=[]
for file in files:
    if file.startswith('.') or file.startswith('_'):
        continue
    else:
        classes_list.append(os.path.join(directory,file))
        audio_classes.append(file)




audio_path_list=[]
for audio_dir in classes_list:
    class_files = os.listdir(audio_dir)
    for file_name in class_files:
        file_path = os.path.join(audio_dir, file_name)
        audio_path_list.append(file_path)




import os

def count_files(directory):
    files_count = {}
    for root, dirs, files in os.walk(directory):
        files_count[root] = len(files)
    return files_count

directory_path = '/Users/aman/Downloads/SpeechCommand'

files_count = count_files(directory_path)

for directory, count in files_count.items():
    print(f"Directory: {directory} - Number of Files: {count}")





import librosa






import librosa
import numpy as np

for i in audio_path_list:
    audio_path = i
    audio, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # print(mfccs)

    mfccs_transposed = mfccs.T

    gradient = np.gradient(mfccs_transposed)




TARGET_SAMPLE_RATE = 15000
MFCC_COMPONENTS = 20
AUDIO_DATASET_PATH = '/Users/aman/Downloads/SpeechCommand'
AUDIO_LABELS = [
    'right', 'eight', 'cat', 'tree', 'bed', 'happy', 'go', 'dog', 'no', 'wow',
    'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero', 'seven', 
    'up', 'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 'five', 'off', 
    'four', '_background_noise_'
]


def compute_mfcc(audio_data, sr):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=MFCC_COMPONENTS)
    return mfcc_features.T

def normalize_features(feature_matrix):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    return feature_scaler.fit_transform(feature_matrix)

def process_single_audio(audio_path):
    raw_audio, original_sr = librosa.load(audio_path, sr=None)
    resampled_audio = librosa.resample(raw_audio, orig_sr=original_sr, target_sr=TARGET_SAMPLE_RATE)

    mfcc_transposed = compute_mfcc(resampled_audio, TARGET_SAMPLE_RATE)
    scaled_mfcc = normalize_features(mfcc_transposed)

    first_derivatives = np.gradient(scaled_mfcc, axis=0)
    second_derivatives = np.gradient(first_derivatives, axis=0)
    scaled_first_derivatives = normalize_features(first_derivatives)
    scaled_second_derivatives = normalize_features(second_derivatives)

    return np.hstack((scaled_mfcc, scaled_first_derivatives, scaled_second_derivatives))

feature_list, label_list = [], []

for current_label in AUDIO_LABELS:
    if current_label == "_background_noise_":
        continue

    directory_path = os.path.join(AUDIO_DATASET_PATH, current_label)    
    for audio_file in os.listdir(directory_path):
        if audio_file.endswith('.wav'):    
            audio_file_path = os.path.join(directory_path, audio_file)
            audio_features = process_single_audio(audio_file_path)
            feature_list.extend(audio_features)
            label_list.extend([current_label] * len(audio_features))


feature_names = [f'mfcc_feature_{i+1}' for i in range(MFCC_COMPONENTS)] + \
                [f'delta_feature_{i+1}' for i in range(MFCC_COMPONENTS)] + \
                [f'delta2_feature_{i+1}' for i in range(MFCC_COMPONENTS)]
audio_feature_df = pd.DataFrame(feature_list, columns=feature_names)
audio_feature_df['audio_label'] = label_list

# print(audio_feature_df)




delta2_columns = [f'delta2_feature_{i+1}' for i in range(MFCC_COMPONENTS)]
audio_feature_df = audio_feature_df.drop(columns=delta2_columns)




audio_feature_df




df = audio_feature_df.copy()
df




mfcc_gmm_models = {}

label_feature_map = {}

grouped_by_label = df.groupby(['audio_label'])

for label_name, features_group in grouped_by_label:
    

    label_feature_map[label_name] = features_group.copy()

    features_group.drop(columns=['audio_label'],axis=1,inplace=True)

    gmm_model = GaussianMixture(n_components=1, covariance_type='full', 
                                random_state=42, max_iter=100,
                                init_params='kmeans')
    
    gmm_model.fit(features_group)

    mfcc_gmm_models[label_name[0]] = gmm_model




def calculate_mfccs(audio, sample_rate, n_mfcc=20):
    return librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

def calculate_gradients(mfccs_transposed):
    return np.gradient(mfccs_transposed, axis=0)

def scale_features(features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(features)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=15000)
    
    mfccs = calculate_mfccs(audio, sample_rate)
    mfccs_transposed = mfccs.T
    
    gradients = calculate_gradients(mfccs_transposed)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    mfccs_scaled = scaler.fit_transform(mfccs_transposed)
    gradients_scaled = scaler.fit_transform(gradients)
    
    features = np.hstack((mfccs_scaled, gradients_scaled))
    
    return features

id_to_label = {
    0: 'right', 1: 'eight', 2: 'cat', 3: 'tree', 4: 'bed', 5: 'happy', 6: 'go', 7: 'dog', 8: 'no', 9: 'wow',
    10: 'nine', 11: 'left', 12: 'stop', 13: 'three', 14: 'sheila', 15: 'one', 16: 'bird', 17: 'zero', 18: 'seven', 19: 'up',
    20: 'marvin', 21: 'two', 22: 'house', 23: 'down', 24: 'six', 25: 'yes', 26: 'on', 27: 'five', 28: 'off', 29: 'four'
}

label_to_id = {label: idx for idx, label in id_to_label.items()}

ordered_gmm_models = [mfcc_gmm_models[id_to_label[i]] for i in range(len(id_to_label))]

def get_prediction(models, feature_vector):
    log_likelihoods = np.array([model.score(feature_vector) for model in models])
    predicted_id = np.argmax(log_likelihoods)

    return predicted_id

audio_id_list = []
predicted_labels_list = []

test_dataset = pd.read_csv("test.csv")

for index, record in test_dataset.iterrows():
    
    audio_id = record["ID"]
    audio_path = "/Users/aman/Kaggle_two/SpeechCommandTest/" + record["AUDIO_FILE"]
    
    audio_features = extract_features(audio_path)

    
    predicted_label = get_prediction(ordered_gmm_models, audio_features)
    
    predicted_labels_list.append(predicted_label)
    audio_id_list.append(audio_id)

results_df = pd.DataFrame({'ID': audio_id_list, 'TARGET': predicted_labels_list})
results_df.to_csv('submission.csv', index=False)


# In[48]:


def predict(path):
    
    
    def calculate_mfccs(audio, sample_rate, n_mfcc=20):
        return librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    def calculate_gradients(mfccs_transposed):
        return np.gradient(mfccs_transposed, axis=0)

    def scale_features(features):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(features)

    def extract_features(file_path):
        audio, sample_rate = librosa.load(file_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=15000)
        
        mfccs = calculate_mfccs(audio, sample_rate)
        mfccs_transposed = mfccs.T
        
        gradients = calculate_gradients(mfccs_transposed)
        
        scaler = MinMaxScaler(feature_range=(0,1))
        mfccs_scaled = scaler.fit_transform(mfccs_transposed)
        gradients_scaled = scaler.fit_transform(gradients)
        
        features = np.hstack((mfccs_scaled, gradients_scaled))
        
        return features
    
    
    id_to_label = {
        0: 'right', 1: 'eight', 2: 'cat', 3: 'tree', 4: 'bed', 5: 'happy', 6: 'go', 7: 'dog', 8: 'no', 9: 'wow',
        10: 'nine', 11: 'left', 12: 'stop', 13: 'three', 14: 'sheila', 15: 'one', 16: 'bird', 17: 'zero', 18: 'seven', 19: 'up',
        20: 'marvin', 21: 'two', 22: 'house', 23: 'down', 24: 'six', 25: 'yes', 26: 'on', 27: 'five', 28: 'off', 29: 'four'
    }

    label_to_id = {label: idx for idx, label in id_to_label.items()}

    ordered_gmm_models = [mfcc_gmm_models[id_to_label[i]] for i in range(len(id_to_label))]

    def get_prediction(models, feature_vector):
        log_likelihoods = np.array([model.score(feature_vector) for model in models])
        predicted_id = np.argmax(log_likelihoods)

        return predicted_id
    
    
    path = path
    
    audio_features = extract_features(audio_path)
    
    predicted_label = get_prediction(ordered_gmm_models, audio_features)
    
    
    return predicted_label

