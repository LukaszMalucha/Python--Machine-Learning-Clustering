## K-Means Clustering 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


## Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []    ## initialize the list
for i in range(1, 11):
        kmeans = KMeans(n_clusters = i,        ## from 1 to 10
                        init = 'k-means++',    ## k-means++ to avoid random initialziation trap
                        max_iter = 300,        ## 300 is deafault        
                        n_init  = 10,          ## algorithm runs with different initial centroids      
                        random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)           ## to compute wcss   
## Createing a plot:

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

## Applying k-means to the mall dataset - from the plot we can see that optimum is 5 clusters.
kmeans = KMeans(n_clusters = 5,
                init = 'k-means++',    ## k-means++ to avoid random initialziation trap
                max_iter = 300,        ## 300 is deafault        
                n_init  = 10,          ## algorithm runs with different initial centroids      
                random_state = 0)
y_kmeans = kmeans.fit_predict(X)       ## fit_predict returns a cluster for each observation 

## Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],   ## specify that we want first cluster + first column vs second column for 'y'
            s = 100, c = 'red',label = 'Careful')                            ## size for datapoints/color
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s = 100, c = 'blue',label = 'Standard') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 100, c = 'green',label = 'Target') 
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],s = 100, c = 'cyan',label = 'Careless') 
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],s = 100, c = 'magenta',label = 'Sensible') 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],         ## cluster centers coordinates
            s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()






                     