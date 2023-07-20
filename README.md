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
The default value for **max_depth** is None, which means that each tree will expand until every leaf is pure. A **pure leaf** is one where all of the data on the leaf comes from the same class.
```ruby
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,Y_train)
```
<img src="/readme_images/1.png">

The **precision** of this model on the **training data** is 100%, but on the **test data**, it drops significantly to 57.9%. This indicates that the model is **overfitting**. 
Additionally, the tree mentioned above lacks **interpretability**.
## max_depth = 2

```ruby
clf = DecisionTreeClassifier(criterion='entropy',max_depth=2)
clf.fit(X_train,Y_train)
```
<img src="/readme_images/2.png">

When max_depth = 2, the precision on the training data is 85.5%, and on the test data, it is 69.2%. Although the training precision has decreased, the **test precision increased**, indicating an improvement in the model's performance. However, there is still room for further improvement. 

Moreover, the tree mentioned above is much more **interpretable** due to its limited depth and simple structure, making it easier to **understand** and **analyze**.
## Hyperparameter Tuning
```ruby
from sklearn.model_selection import RandomizedSearchCV
```
```ruby
param_dist = {'max_depth': np.arange(1,15,1)}
clf = DecisionTreeClassifier(criterion='entropy')
rand_search = RandomizedSearchCV(clf, param_distributions = param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, Y_train)
# Create a variable for the best model
best_clf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best Hyperparameters:',  rand_search.best_params_)
```
| Confusion Matrix  | Classification Report |
| --- | --- |
| <img src="/readme_images/c.png"> | <img src="/readme_images/r.jpg"> |

<img src="/readme_images/3.png">
