# Visualization tool
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df
Y_true = np.asarray(test_res['Survived'])

##------------------- Logistic Regression----------------------------
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
f1_log = metrics.f1_score(Y_true, Y_pred)
preci_log = metrics.precision_score(Y_true, Y_pred)
recall_log = metrics.recall_score(Y_true, Y_pred)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# See the importance of features
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Importance"] = pd.Series(logreg.coef_[0])
coeff_df["Importance%"] = 100.0 * (abs(coeff_df["Importance"]) \
        / abs(coeff_df["Importance"]).sum())
coeff_df.sort_values(by='Importance%', ascending=False)

##------------------------------ SVC ---------------------------------
##-------------------- Support Vector Machines -----------------------
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
f1_svc = metrics.f1_score(Y_true, Y_pred)
preci_svc = metrics.precision_score(Y_true, Y_pred)
recall_svc = metrics.recall_score(Y_true, Y_pred)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

##---------------------------- K-nn ----------------------------------
##------------------k-Nearest Neighbors algorithm --------------------
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
f1_knn = metrics.f1_score(Y_true, Y_pred)
preci_knn = metrics.precision_score(Y_true, Y_pred)
recall_knn = metrics.recall_score(Y_true, Y_pred)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

##---------------------- Decision Tree -------------------------------
decision_tree = DecisionTreeClassifier()
mytree = decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
f1_decision_tree = metrics.f1_score(Y_true, Y_pred)
preci_decision_tree = metrics.precision_score(Y_true, Y_pred)
recall_decision_tree = metrics.recall_score(Y_true, Y_pred)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

from sklearn import tree
with open("treeTitanic.dot","w") as f:
    f = tree.export_graphviz(mytree,out_file=f)


from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
tree.export_graphviz(decision_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

##---------------------- Random Forest -------------------------------
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
f1_random_forest = metrics.f1_score(Y_true, Y_pred)
preci_random_forest = metrics.precision_score(Y_true, Y_pred)
recall_random_forest = metrics.recall_score(Y_true, Y_pred)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


##*********************** Summary of results ************************
# In a binary classification task, the terms "positive" and "negative"
# refer to the classifier's prediction, and the terms "true" and "false" 
# refer to whether that prediction corresponds to the external judgment 
# (sometimes known as the "observation").  
# ********************************************************************
## See the summary of results
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree'],
    'F_score': [f1_svc, f1_knn, f1_log, 
                f1_random_forest, f1_decision_tree],
    'Precesion_score': [preci_svc, preci_knn, preci_log, 
                preci_random_forest, preci_decision_tree],
    'Recall_score': [recall_svc, recall_knn, recall_log, 
                recall_random_forest, recall_decision_tree]
#    'Score': [acc_svc, acc_knn, acc_log, 
#              acc_random_forest, acc_decision_tree],
                 })
