import pandas as pd

def load_and_filter_data(path='data/EEG_data.csv'):
    input_df = pd.read_csv(path)
    ptsd = input_df[input_df['specific.disorder'] == 'Posttraumatic stress disorder']
    hc = input_df[input_df['specific.disorder'] == 'Healthy control'].sample(n=52, random_state=42)
    data = pd.concat([ptsd, hc])
    data.drop(columns=['no.', 'eeg.date', 'main.disorder', 'Unnamed: 122', 'age', 'education', 'sex'], inplace=True)
    data['specific.disorder'] = data['specific.disorder'].map({'Healthy control': 0, 'Posttraumatic stress disorder': 1})
    return data
