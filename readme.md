## Implementation K-medoids algorithm with selection of the second best point as an medoids 

### Task Description
The aim of this project is to implement k-medoids clustering algorithm and test it on real data. The algorithm should be able to operate on every amount of attributes and its final output should be similar to the output of sklearn implementation. 

#### Agenda
1. Algorithm description
2. Implementation
3. Testing on mocked data
4. Testing time complexity
5. Testing on real data
6. Summary 

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

#### 2. Implementation

#### 3. Testing on Mocked Dataset
Steps for creating mocked dataset:
- Initialize k means, for this case points (0, 0), (10, 5), (0, 10) were chosen
- For each cluster create N = 100 multivariate normal points which covariance equals to [ [2,0] [0,2] ]
- Concanate members of each cluster into one dataset (total size of dataset is 3 * N x 2)
- create 3 * N x 1 array with labels to compare result of clastering with actual clusters

So the final input of the training dataset is an array of dim 3*N x 2

The sample dataset and its visualization can be seen at the pictures:
<p align="center">
  <img src = "https://imgur.com/W6q29WS.png"/>
</p>

And at the picture above the result of the first clustering can be seen:
<p align="center">
  <img src = "https://imgur.com/Mj20djM.png"/>
</p>

As can be seen, the algorithm works perfectly on the simple, training dataset.
