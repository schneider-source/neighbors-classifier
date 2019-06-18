# neighbors-classifier

Fast, simple, but well-performing classifier based on weighted distances between 
data points. 

## Prerequisites

* python >= 3.7.3
* numpy >= 1.17.0
* pandas >= 0.24.2
* sklearn >= 0.21.2
* matplotlib >= 3.0.3

## Installing

I recommend downloading the newest 
[anaconda3](https://www.anaconda.com/distribution/#download-section) version. 
Then install the following packages e.g. via

```
pip install numpy pandas sklearn matplotlib
```

Clone the repository e.g. with 

```
git clone https://github.com/schneider-source/neighbors-classifier.git
```

## Running example

An example of how to use the classifier and a performance comparison with 
[sklearn.neighbors.RadiusNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)
using the [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) is provided. 
First, cd into the neighbors-classifier directory. Then execute the script via

```
python3 example.py
```

from the command line or

```
%run example
```

from inside an IPython session. Follow the console output for performance scores
and runtime comparison, which should give you something similar to the following. 

```
Neighbors classifier example and comparison with 
sklearn.neighbors.RadiusNeighborsClassifier

PCA: EVR = [0.84136038 0.11751808], SUM = 0.9588784639918416

Test performance of classifiers for full dataset

NeighborsClassifier:
accuracy_score    0.9452

sklearn.neighbors.RadiusNeighborsClassifier:
accuracy_score    0.888

For the confusion matrices see figure images/full_dataset/cm.png.

My NeighborsClassifier should have slightly better performance
than sklearn.neighbors.RadiusNeighborsClassifier.

For feature space classification see figure images/full_dataset/classification.png.

Test performance of classifiers for asymmetric data
(using no setosa, all 50 versicolor, and only 20 versicolor samples)

NeighborsClassifier:
accuracy_score    0.943333

sklearn.neighbors.RadiusNeighborsClassifier:
accuracy_score    0.774167

For the confusion matrices see figure images/asymmetric/cm.png.

My NeighborsClassifier should have significantly better performance
than sklearn.neighbors.RadiusNeighborsClassifier.

For feature space classification see figure images/asymmetric/classification.png.

Compare execution time of classifiers

Runtime NeighborsClassifier = 1.047 sec
Runtime sklearn.neighbors.RadiusNeighborsClassifier = 10.793 sec
My classifier is 10.31 x faster than the sklearn classifier.
```

More importantly, look for the generated figures in directory 'images', 
which should look similar to the following.

#### Full dataset

Using all 50 samples from each of three species of Iris 
(Iris setosa, Iris virginica and Iris versicolor).

![alt text](https://github.com/schneider-source/neighbors-classifier/blob/master/images/full_dataset/cm.png)
![alt text](https://github.com/schneider-source/neighbors-classifier/blob/master/images/full_dataset/classification.png)

#### Asymmetric label frequency

Using no Iris setosa, all 50 Iris versicolor, and only the first 20 Iris versicolor 
samples.

![alt text](https://github.com/schneider-source/neighbors-classifier/blob/master/images/asymmetric/cm.png)
![alt text](https://github.com/schneider-source/neighbors-classifier/blob/master/images/asymmetric/classification.png)

## License

See the [LICENSE](LICENSE) for details.


