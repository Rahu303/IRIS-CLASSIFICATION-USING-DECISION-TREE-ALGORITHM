from sklearn.datasets import load_iris
iris = load_iris()
x =iris.data
x.shape
y=iris.target
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.tree import plot_tree
plot_tree(clf)