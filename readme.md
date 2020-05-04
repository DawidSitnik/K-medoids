## Implementation K-medoids algorithm with selection of the second best point as an medoids 

### Task Description
The aim of this project is to implement k-medoids clustering algorithm and test it on real data. The algorithm should be able to operate on every amount of attributes and its final output should be similar to the output of sklearn implementation. 

#### Agenda
1. Algorithm description
2. Testing on mocked data
3. Testing time complexity
4. Testing on real data
5. Summary 

#### 1. Algorithm Description
K-medoids is a partitional clustering algorithm which is modified version of K-means algorithm. Both of them aims to minimizing squared error, however K-medoids is more robust to noise. In K-medoids data points are chosen to be medoids, but in K-means the means are chosen as centroids. 

The difference between those two algorithms can be compared to a difference between mean and median. Mean indicates the average value of all points in the cluster, while median indicates the value around which all data items are evenly distributed around. The main idea behind K-medoids is to compute points which represents each cluster (medoids) and attach each point from the space to the nearest one.

The whole process consist of two steps:
- Randomly selecting k points which represents each cluster.
- Randomly selecting new medoids as long as the mean error decreases 

Algorithm is as follows:
- Select k initial points
- Attach each point from the space to the nearest medoids
- For each non-selected point try to set it as medoid and check wether mean squared error decreases
- Repeat step 2 and 3 until there is no change in medoids

Adventage of K-medoids is its simplicity. It is not the most efficient algorithm however and is mostly used with small datasets. It's time complexity is O(k(n-k)^2).

#### Testing on Mocked Dataset
Steps for creating mocked dataset:
- Initialize k means, for this case points (0, 0), (10, 5), (0, 10) were chosen
- For each cluster create N = 100 multivariate normal points which covariance equals to [ [2,0] [0,2] ]
- Concanate members of each cluster into one dataset assigning them class number of each cluster
