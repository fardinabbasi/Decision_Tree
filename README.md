# Decision Tree
Performing a Decision Tree classifier on the '[Diabetes.csv](https://github.com/fardinabbasi/Decision_Tree/blob/main/Diabetes.csv)' dataset to distinguish between diabetes and non-diabetes cases.

The dataset is divided into a **training set** and a **test set** during the **preprocessing** step.

The scikit-learn **built-in function** is utilized for implementing the Decision Tree classifier.
```ruby
from sklearn.tree import DecisionTreeClassifier
```
In every machine learning model, it is essential to carefully select **hyperparameters**.

Decision tree models have various hyperparameters, including but not limited to **criterion**, **max_depth**, **min_samples_split**, and **max_leaf_nodes**. Let's examine the consequences of choosing the **max_depth** incorrectly:
## No max_depth Limitation
The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
```ruby
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,Y_train)
```
<img src="/readme_images/1.png">

The precision of this model on the training data is 100%, but on the test data, it drops significantly to 57.9%. This indicates that the model is **overfitting**. 
Additionally, the tree mentioned above lacks **interpretability**.
## max_depth = 2

```ruby
clf = DecisionTreeClassifier(criterion='entropy',max_depth=2)
clf.fit(X_train,Y_train)
```
