import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    DATAPATH = '../data/mushrooms.csv'

    data = pd.read_csv(DATAPATH)

    encoded_data = pd.get_dummies(data)
    print(encoded_data.head(5))
    encoded_data.to_csv('../data/mushrooms_encoded.csv', index=False)

main()
