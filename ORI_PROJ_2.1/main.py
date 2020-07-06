import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
pd.set_option('display.max_columns', 25)

# Read csv file into dataframe object
credit_cards = pd.read_csv('credit_card_data.csv')
credit_cards = credit_cards.dropna(how='any', axis=0)

# Print first five rows
print(credit_cards.head())

# From the following line, we can see that features are on different scale
print(credit_cards.max())

# Initialize the Scaler object that scales all features to [0,1] of the dataframe. Why? In order for all of them to have equal
# importance during clusterization (during calculating pairwise distances)
mms = MinMaxScaler()

# Transform all features except CUST_ID because it is a string
features_scaled = mms.fit_transform(credit_cards.iloc[:,1:])

# Replace all transformed features with their scaled versions
credit_cards.iloc[:,1:] = features_scaled

# Print first five rows to see the transformation
print(credit_cards.head())

# From this for loop, we determine the optimal number of clusters by the so-called elbow method
sum_of_squared_dist = []
ks = range(1,15)
for k in ks:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(features_scaled)
    # km.inertia provides average distances from points to centers of clusters.
    sum_of_squared_dist.append(kmeans.inertia_)

# Plot the list of number of clusters against sum_of_squared_dist
plt.plot(ks, sum_of_squared_dist, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.show()


kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(features_scaled)

#
credit_cards['CLUSTER_ID'] = y_kmeans
print(credit_cards.head())

features_inversed = mms.inverse_transform(features_scaled)
credit_cards.iloc[:, 1: -1] = features_inversed
cluster_1 = credit_cards[credit_cards.CLUSTER_ID == 0]
cluster_2 = credit_cards[credit_cards.CLUSTER_ID == 1]
cluster_3 = credit_cards[credit_cards.CLUSTER_ID == 2]

print('############## CLUSTER 1 ##############')
print(cluster_1.describe())

print('############## CLUSTER 2 ##############')
print(cluster_2.describe())

print('############## CLUSTER 3 ##############')
print(cluster_3.describe())

pca = PCA(n_components=2)
data_in_2d = pca.fit_transform(features_inversed)
credit_cards_2d = pd.DataFrame()
color_map = {0 : 'r', 1 : 'b', 2 : 'g'}
label_color = [color_map[i] for i in y_kmeans]
plt.scatter(data_in_2d[:,0], data_in_2d[:,1], c=label_color)
plt.show()