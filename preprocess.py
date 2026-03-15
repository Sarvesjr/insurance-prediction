import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.drop(['id'], axis=1)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    df['Vehicle_Age'] = df['Vehicle_Age'].map({
        '< 1 Year': 0,
        '1-2 Year': 1,
        '> 2 Years': 2
    })
    return df