# Implementation K-Medoids Algorithm With a Selection of the Second-Best Point as a Medoids 

## Task Description
This project aims to implement a k-medoids clustering algorithm and test it on real data. The algorithm should be able to operate on every amount of attributes and its final output should be similar to the output of sklearn implementation. 

### Table of Contents
1. Algorithm description
2. Implementation
3. Testing on mocked data
4. Testing on real data
5. Summary 

### 1. Algorithm Description
K-medoids is a partitional clustering algorithm that is a modified version of the K-means algorithm. Both of them aim to minimize squared error, however, K-medoids is more robust to noise. In K-medoids data points are chosen to be medoids, but in K-means the means are chosen as centroids. 

The difference between those two algorithms can be compared to a difference between mean and median. Mean indicates the average value of all points in the cluster, while the median indicates the value around which all data items are evenly distributed around. The main idea behind K-medoids is to compute points that represent each cluster (medoids) and attach each point from the space to the nearest one.

The whole process consists of two steps:
- Randomly selecting k points that represent each cluster.
- Randomly selecting new medoids as long as the mean error decreases 

The algorithm is as follows:
- Select k initial points
- Attach each point from the space to the nearest medoids
- For each non-selected point try to set it as medoid and check whether mean squared error decreases
- Repeat step 2 and 3 until there is no change in medoids

The advantage of K-medoids is its simplicity. It is not the most efficient algorithm however and is mostly used with small datasets. Its time complexity is O(k(n-k)^2).

### 2. Implementation
The whole program was implemented in Python which was chosen because of its syntax simplicity and machine learning related libraries. The code could be split into 3 parts: K-medoids implementation, testing on mocked data, and comparison with other algorithms on real data. 
All of them were created in the form of a script that executes sequentially and there is no complicated object structure behind it. In the above section, I will describe all three groups.

#### K-medoids Implementation
This section consists of class k_medoids and one helper function for calculating Euclidean distance between two points. K_medoids class has four attributes:

- **k** - number of clusters, the value passed by parameter while object creation
- **max_iter** - maximal number of iterations of the algorithm, the value passed by parameter while object creation
- **is_converged** - a flag that decides weather the algorithm converged. Initially set to 0, set its value to 1 if any other k-medoid gives a better result than current.
- **medoids_cost** - list of distances from the nearest medoids, initially set to empty list. It is cleared every each iteration.

and five functions:

- **initMedoids(self, X)** - chooses K random medoids from dataset X, which is passed by a parameter.
- **is converged(self, new_medoids)** - checks whether new_medoids are the same as the current.
- **updateMedoids(self, X, labels)** - checks weather new medoids gives smaller mean error than the current ones. If so, it set new medoids as medoids. If current medoids have been giving the best possible solution it sets is_converged flag to 1 and breaks the loop in *fit()* function. Takes dataset X and labels as the arguments
- **fit(self, X)** - generate new medoids, assigns each point to them basing on min euclidian distance, and try to update the model. It runs in the loop for *max_iter* times, breaks when is_converged is 1. 
- **predict(self, data)** - calculates the distance from each point in which class is supposed to be predicted to each medoid and assign it to the nearest one. 

#### Testing on Mocked Data
This part creates a dataset with mocked data which serves for checking whether the algorithm works properly on simple data. It visualizes created datapoints with its actual classes and predicted ones. Then, the algorithm is executed many times on different sizes of testing data to checks time dependence on dataset size. 

#### Comparision with Other Algorithms on Real Data
In this part, the algorithm is tested on *Iris Dataset* and compared with the implementation of K-medoid from *sklearn library* and with *K-means* algorithm from the same library.

### 3. Testing on Mocked Dataset
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

The learning time based on size of the training dataset can be seen at the picture below:

#### Time Complexity

Despite some iterations in which initial medoids were extraordonary unfavourable, the graph is similar to x squared function. The assumption about algorithm complexity was correct and this type of algorithm is not the best choice for big datasets. 

<p align="center">
  <img src = "https://imgur.com/0dyJLPv.png"/>
</p>

### 4. Testing on Real Data
In this part, the algorithm will be used for clustering on *Iris Dataset* and compared with the implementation of K-means from *Sklearn library* and with the K-means algorithm from the same library.

**Iris Dataset** consists of 150 samples with four attributes: sepal length, sepal width, petal length, and petal width. Each sample has one of 3 labels (from 0 to 2). The dataset doesn't consist of any outliers. 

The visualisation of dataset can be seen at the picture below:
<p align="center">
  <img src = "https://imgur.com/D3zRaxA.png"/>
</p>

#### Comparison of Algorithms
Before training and testing algorithms, the dataset was split into training and testing sets in a 100:50 ratio.
Each point was assigned to the set only once and was chosen randomly. The final input for the algorithms was an array of float numbers with dimensionality 100X4. Then, to predict labels for testing dataset the array of 50x4 was passed to predict the function of each model.

Because of the small size of the dataset, the cross-validation was used, which means that the testing phase was executed 200 times, and the accuracy and diff values are mean values from those iterations.

The graphical representation of assigment from one iteration of cross-validation can be seen at the picture below:
<p align="center">
  <img src = "https://imgur.com/ELjiikE.png"/>
</p>


The final result can be seen below:


|                  | Accuracy, test     | Accuracy, train | Differences, test | Differences, train |
|------------------|--------------------|----------------|--------------------|--------------------|
| My K-medoid      |        62.42%      |     71.92%     |        18.79       |        14.04       |
| Sklearn K-medoid |        68.00%      |     80.00%     |        16.00       |        10.00       |
| Sklearn K-means  |        73.98%      |     88.88%     |        13.01       |        5.56        |

What could be seen in the table is that my implementation of K-medoid is just 5.5% worse on testing and 8% on the training dataset than original sklearn implementation on average. The mean value of differences for my algorithm is 19 and 14 for testing and training data consequently, while for SKlearn those numbers are 16 and 10. The highest score was achieved by Sklearn K_means, which got 73% on testing, 89% on training data and have made just 13 and 5.5 mistakes for the same data consequently.  

#### Algorithms Robustness for Outlayers
The last aspect that was checked was robustness for outlayers. Due to that 3 ponts with extremly high values were added to the training dataset, which visualization is as follows:
<p align="center">
  <img src = "https://imgur.com/SPp11Qu.png"/>
</p>

|                  | Accuracy, test     | Accuracy, train | Differences, test | Differences, train |
|------------------|--------------------|----------------|--------------------|--------------------|
| My K-medoid      |        61.22%      |     74.22%     |        19.39       |        17.14       |
| Sklearn K-medoid |        64.00%      |     76.00%     |        18.00       |        12.00       |
| Sklearn K-means  |        38.12%      |     41.28%     |        30.12       |        29.23       |

As we assumed at the beginning, K-medoids is robust for outliers, his scores didn't get worse too much comparing to the previous one. K-means, however, were significantly worse and is outlier sensitive. 

### 5. Summary
After the experiment, all of the assumptions stated were meet, so:
- the time complexity increases in O(N^2)
- the algorithm is outlier robust
- the final result was almost as good as the one from the most popular libraries

The obtained result could be much better if I would have chosen a larger dataset. 100 training examples, while taking 50 for testing isn't the most beneficial ratio for the algorithm. 

In the future, to make the algorithm work faster I could try to reduce the time needed for getting into convergence state by changing how initial medoids are chosen. For now, there are few algorithms for this time. The one which could be tried as the first would be k-means++ which is used while initialization of central points in a k-means algorithm.
