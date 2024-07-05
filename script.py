from Kaggle_2 import predict
import os
import pandas as pd

path = 'wav_file_paths.txt'

with open(path, 'r') as file:
    audio_paths = file.readlines()

audio_paths = [path.strip() for path in audio_paths]

idlist = []
predlist = []

for idx, path in enumerate(audio_paths):
    prediction = predict(path)
    predlist.append(prediction)
    idlist.append(idx)


submission_df = pd.DataFrame({'ID': idlist, 'TARGET': predlist})
submission_df.to_csv('submission.csv', index=False)

print("CSV file created successfully.")