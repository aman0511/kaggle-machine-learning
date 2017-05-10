import pandas
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import time

train = pandas.read_csv("kaggle-mnist-train.csv")

#NUM_TRAIN_IMAGES = 2000
#X = train.iloc[0:NUM_TRAIN_IMAGES,1:]
#Y = train.iloc[0:NUM_TRAIN_IMAGES,:1].values.ravel()
#X[X>0]=1

#scores = cross_val_score(svm.SVC(), X, Y, cv=5)
#print "Average score: %f" % np.mean(scores)

# Train on the training data..
#X = train.iloc[:,1:]
#Y = train.iloc[:,:1]
#X[X>0]=1

train_image = train.ix[:,1:]
train_label = train.ix[:,0]
X = train_image.values / 255.0
Y = train_label.values

# X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=0.8, random_state=0)
#
#
# def n_component_analysis(n, X_train, y_train, X_val, y_val):
#     start = time.time()
#
#     print("PCA begin with n_components: {}".format(n))
#     pca = PCA(n_components=n)
#     pca.fit(X_train)
#
#     X_train_pca = pca.transform(X_train)
#     X_val_pca = pca.transform(X_val)
#
#     print('SVC begin')
#     clf1 = svm.SVC()
#     clf1.fit(X_train_pca, y_train)
#
#     accuracy = clf1.score(X_val_pca, y_val)
#
#     end = time.time()
#     print("accuracy: {}, time elaps:{}".format(accuracy, int(end-start)))
#
#     return accuracy
#
# n_s = np.linspace(0.70, 0.85, num=15)
# accuracy = []
# for n in n_s:
#     tmp = n_component_analysis(n, X_train, y_train, X_val, y_val)
#     accuracy.append(tmp)
#
#
# import matplotlib.pyplot as plt
# plt.plot(n_s, np.array(accuracy), 'b-')
# plt.show()
#
# exit()

print "Starting pca"

pca = PCA(n_components=0.74)
pca.fit(X)

X = pca.transform(X)

print "Done with pca"

# gsc = GridSearchCV(
#     estimator=svm.SVC(),
#     param_grid={
#         'C': [0.1, 0.5, 1, 2, 3, 4]
#     },
#     cv=3,
#     verbose=5
# )
# gsc.fit(X, Y)
# print gsc.cv_results_
# exit()


classifier = svm.SVC(C=4)
classifier.fit(X, Y)

# Predict numbers
test = pandas.read_csv("kaggle-mnist-test.csv")
#test = test.iloc[:,:]
#test[test>0]=1
test = test.ix[:,:]

test = pca.transform(test.values / 255.0)

predictions = classifier.predict(test)
predictions = pandas.DataFrame(predictions)
predictions.index.name='ImageId'
predictions.index+=1
predictions.columns=['Label']
predictions.to_csv('kaggle-mnist-predictions.csv', header=True)
