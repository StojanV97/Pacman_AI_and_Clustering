import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

def load_data():
    df = pd.read_csv(sys.argv[2])
    return df


if __name__ == '__main__':
    df = load_data()

    # Izbacujemo CUST_ID kolonu, jer sadrzi tekstualne vrednosti koje nisu od znacaja za model
    df = df.drop("CUST_ID", axis='columns')

    # Funkcija koja izbacuje sve redove koje sadrze NaN vrednost u nekom polju
    df = df.dropna()

    # -------------------------------------- DATA LOADED FULLY ---------------------------------------- #


    #scaler = StandardScaler()
    #df_norm = pd.DataFrame(scaler.fit_transform(df.astype(float)))


    #kmeans = KMeans(n_clusters=2, max_iter=100)
    #kmeans.fit(df)

    # type(balance) = pandas.core.series.Series
    balance = df["BALANCE"]
    purchases = df["PURCHASES"]

    plt.scatter(balance, purchases)
    plt.show()