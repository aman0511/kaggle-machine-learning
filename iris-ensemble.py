from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

X,y = datasets.load_iris(return_X_y=True)

svc = SVC(C=1.5)
lr = LogisticRegression(C=65.0, solver='lbfgs', multi_class='multinomial')
tree = DecisionTreeClassifier()
rf = RandomForestClassifier(min_samples_leaf=10, min_samples_split=10)
knn = KNeighborsClassifier(n_neighbors=12, weights='distance')

#ensemble = EnsembleVoteClassifier([svc, lr, rf, knn, tree])
ensemble = EnsembleVoteClassifier([svc, lr, knn])

# print "SVC                Score : %.4f" % cross_val_score(svc, X, y, cv=5).mean()
# print "LogisticRegression Score : %.4f" % cross_val_score(lr, X, y, cv=5).mean()
# print "DecisionTree       Score : %.4f" % cross_val_score(tree, X, y, cv=5).mean()
# print "RandomForest       Score : %.4f" % cross_val_score(rf, X, y, cv=5).mean()
# print "KNN                Score : %.4f" % cross_val_score(knn, X, y, cv=5).mean()
# print "EnsembleVote       Score : %.4f" % cross_val_score(ensemble, X, y, cv=5).mean()
#
# svc.fit(X,y)
# plot_confusion_matrix(confusion_matrix(y, svc.predict(X)))
# plt.show()
#
# exit()

X = X[:,[0, 2]]

gs = gridspec.GridSpec(2, 3)
fig = plt.figure(figsize=(10, 8))

labels = [
    'SVC',
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'KNN',
    'Ensemble'
]

for clf, lab, grd in zip([svc, lr, tree, rf, knn, ensemble],
                         labels,
                         itertools.product([0, 1], [0, 1, 2])):
    print "%s Score : %.4f" % (lab, cross_val_score(clf, X, y, cv=5).mean())

    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])
    #fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2, filler_feature_values={2: 2.0, 3: 2.0})
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)

plt.show()
