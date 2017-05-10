import pandas
from sklearn import linear_model, svm, tree, ensemble
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
import numpy as np
import math


data = pandas.read_csv('huizen2.csv')


def check_corellation(col1, col2):
    pr = pearsonr(data.as_matrix([col1]), data.as_matrix([col2]))

    if pr[1] < 0.05:
        print 'Column %s has correlation coeffictient %f with p-value %f' % (col2, pr[0], pr[1])
    else:
        print 'Column %s has no significant correlation' % col2

def correlations():
    check_corellation('Prijs', 'Meters')
    check_corellation('Prijs', 'Kamers')
    check_corellation('Prijs', 'Perceel')
    check_corellation('Prijs', 'Tussenwoning')
    check_corellation('Prijs', 'Hoekwoning')
    check_corellation('Prijs', 'Vrijstaand')
    check_corellation('Prijs', 'Jaren-30')
    check_corellation('Prijs', 'Jaren-50')
    check_corellation('Prijs', 'Jaren-60')
    check_corellation('Prijs', 'Jaren-90')
    check_corellation('Prijs', 'Jaren-2000')
    check_corellation('Prijs', 'Jaren-2010')


#correlations()


def fit_validate(y, x):
    scores = cross_val_score(linear_model.LinearRegression(), x, y, cv=10, scoring='neg_mean_squared_error')
    print "Average score: %f" % math.sqrt(- np.mean(scores))


#fit_validate(data.as_matrix(['Prijs']), data.as_matrix(['Meters']))
#fit_validate(data.as_matrix(['Prijs']), data.as_matrix(['Kamers']))
#fit_validate(data.as_matrix(['Prijs']), data.as_matrix(['Perceel']))
#fit_validate(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel']))
#fit_validate(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel', 'Jaren-90']))
#fit_valDecisionTreeRegressoridate(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel','Tussenwoning','Hoekwoning','Vrijstaand','Jaren-30', 'Jaren-50', 'Jaren-60', 'Jaren-90', 'Jaren-2000', 'Jaren-2010']))


def fit_print_eval(y,x):
    regr = linear_model.Lasso(alpha=0.9, max_iter=10000)
    regr.fit(x, y)

    print ' y = {0} + '.format(regr.intercept_[0]) +\
          ' + '.join(['x{0} * {1}'.format(i,w) for i,w in enumerate(regr.coef_)])

    print regr.score(x, y)

    return regr


#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Kamers']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Perceel']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Perceel']))
#model = fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel']))
#model = fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel','Jaren-90']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Tussenwoning','Hoekwoning','Vrijstaand']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Jaren-30', 'Jaren-50', 'Jaren-60', 'Jaren-90', 'Jaren-2000', 'Jaren-2010']))
#fit_print_eval(data.as_matrix(['Prijs']), data.as_matrix(['Meters','Kamers','Perceel','Tussenwoning','Hoekwoning','Vrijstaand','Jaren-30', 'Jaren-50', 'Jaren-60', 'Jaren-90', 'Jaren-2000', 'Jaren-2010']))
#print 'Price of my house %f' % model.predict([[92,4,92]])[0]


def fit_and_plot():
    y = data.as_matrix(['Prijs'])
    x = data.as_matrix(['Meters'])

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    plt.figure(figsize=(10,6))
    plt.scatter(x, y, color='black')
    plt.plot(x, regr.predict(x), color='blue', linewidth=3)
    plt.xticks(np.arange(x.min(), x.max(), 25))
    plt.xlabel('Floor space')
    plt.yticks(np.arange(y.min(), y.max(), 50000))
    plt.ylabel('Price')
    plt.title('House price vs floor space')
    plt.show()

fit_and_plot()

exit()

models = {
    'Linear Regression': linear_model.LinearRegression(),
    #'ridge': linear_model.Ridge(),
    #'lasso': linear_model.Lasso(),
    'Elastic Net': linear_model.ElasticNet(),
    'SVR': svm.SVR(kernel='linear'),
    'Random Forest': ensemble.RandomForestRegressor()
}

scores_algorithms = {}

for name, model in models.items():
    scores = cross_val_score(model, data.as_matrix(['Meters','Kamers','Perceel']), data.as_matrix(['Prijs']),
                             cv=10, scoring='neg_mean_squared_error')
    #print "Model %s has average score: %f" % (name, math.sqrt(- np.mean(scores)))

    scores_algorithms[name] = math.sqrt(- np.mean(scores))

y_pos = np.arange(len(scores_algorithms))

plt.bar(y_pos, scores_algorithms.values(), align='center', alpha=0.5)
plt.xticks(y_pos, scores_algorithms.keys())
plt.ylabel('Error')
plt.title('Average error per algorithm')

plt.show()