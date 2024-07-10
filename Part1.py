# Data Mining - cluster analysis
import pandas as pd
from sklearn.cluster import KMeans

desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
df = pd.read_csv(r'single_family_home_values.csv')
# zillow file
df2 = df.drop('estimated_value', axis=1)  # any data frame column can be dropped

df3 = df2[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']]  # reduced df
df3.fillna(0, inplace=True)  # replaces the NaN in priorSaleAmount with 0 -- may get a warning, but better than NaN
print(df3.head(2))  # prints top two rows of df3
k_groups = KMeans(n_clusters=5, random_state=0).fit(df3)  # separates data set into 5 distinguishable groups
print(k_groups.labels_)  # displays k_groups' label (0 to 4) for each row
print(len(k_groups.labels_), df3.shape)  # displays rows in k_groups as well as rows, columns in df3
print(k_groups.cluster_centers_)  # displays averages of the seven columns for each cluster centroid [0, 1, 2, 3, 4]
print(k_groups.cluster_centers_[0])  # displays averages for each of the seven columns in the cluster centroid [0]
df3['cluster'] = k_groups.labels_  # add a new column to df3 called 'cluster', the k-group #
print(df3.head(3))  # display the top three rows of data frame df3
print(df3.groupby('cluster').mean())  # display the means of the seven columns of data frame df3
from sklearn.metrics import silhouette_score  # coefficient score where higher is better, 0 = cluster overlap

df4 = df3.drop('cluster', axis=1)  # create a new data frame df4 that dropped the cluster column
# for loop to determine optimum K groups
for i in range(3, 10):  # for loop to determine best number of K clusters between 3 and 10
    k_groups = KMeans(n_clusters=i).fit(df4)  # K clusters must have atleast 2 clusters
    labels = k_groups.labels_
    print('K Groups = ', i, 'Silhouette Coeffient = ', silhouette_score(df4, labels))  # displays i and coefficient
# End of Data Mining - Cluster Analysis
