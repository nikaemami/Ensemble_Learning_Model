# Ensemble_Learning_Model
Implementation of a voting-based ensemble learning model with 3 methods of SVM, Random Forest, and Logistic Regression with K-fold cross-validation in scikit-learn, on a cancer dataset.

A voting classifier is a machine learning estimator that trains various base models or estimators and predicts on the basis of aggregating the findings of each base estimator.

We can implement the voting classifier in 2 ways:

<h4> &nbsp;Majority Voting:</h4>

Every model makes a prediction (votes) for each test instance and the final output prediction is the one that receives **more than half of the votes**. If none of the predictions get more than half of the votes, we may say that the ensemble method could not make a stable prediction for this instance.

<h4> &nbsp;Weighted Voting:</h4>

Unlike majority voting, where each model has the same weights, we can increase the importance of one or more models. In weighted voting we count the prediction of the better models **multiple times**.

So in general, in ensemble methods, instead of learning a weak classifier, we learn **many weak classifiers** that are good at different parts of the input space.

Before implementing the ensemble algorithm, I **preprocess** the data, since there is **missing data** in the columns.
I replace the missing values with the **mean of the column**:

```ruby
data["Bare Nuclei"].replace({"?": Bare_Nuclei_mean}, inplace=True)
```

Then I process the dataset by implementing K-fold cross-validation with k = 10, using  **KFold(n_splits=10)** as below:

```ruby
kf = KFold(n_splits=10)
```

Now that the dataset is ready, I implement the Ensemble Learning algorithm with **Random Forest**, **SVM**, and **Logistic Regression** with voting classifier, using **VotingClassifier** in scikit-learn. 

```ruby
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)])
```

Finally I report the **average accuracy** of the model. The result is as follows:

Avg accuracy: 0.9699792960662528
