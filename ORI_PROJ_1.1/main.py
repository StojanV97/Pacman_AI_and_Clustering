import pandas as pd

def load_data():
    df = pd.read_csv("credit_card_data.csv")
    return df


if __name__ == '__main__':
    df = load_data()

    print(type(df))