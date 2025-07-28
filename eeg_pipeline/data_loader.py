import pandas as pd
from .config import DATA_PATH

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    ptsd = df[df['specific.disorder'] == 'Posttraumatic stress disorder']
    hc = df[df['specific.disorder'] == 'Healthy control'].sample(n=52, random_state=42)
    data = pd.concat([ptsd, hc])

    data.drop(columns=['no.', 'eeg.date', 'main.disorder', 'Unnamed: 122', 'age', 'education', 'sex'], inplace=True)
    data['specific.disorder'] = data['specific.disorder'].map({'Healthy control': 0, 'Posttraumatic stress disorder': 1})

    return data
