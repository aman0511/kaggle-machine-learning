from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt

X,y = datasets.load_diabetes(return_X_y=True)

#X_train, X_test, y_train, y_test = train_test_split(X, y)

def objective_lasso(params):
    clf = Lasso(alpha=params[0])

    #clf.fit(X_train, y_train)
    #mae = mean_absolute_error(y_test, clf.predict(X_test))

    clf.fit(X, y)
    mae = mean_absolute_error(y, clf.predict(X))

    print("Lasso(alpha={}) => Score {}".format(params[0], mae))

    return mae

def objective_svr(params):
    clf = SVR(C=params[0], epsilon=params[1])
    clf.random_state = 12345

    #clf.fit(X_train, y_train)
    #mae = mean_absolute_error(y_test, clf.predict(X_test))

    clf.fit(X, y)
    mae = mean_absolute_error(y, clf.predict(X))

    print("SVR(C={}, epsilon={}) => Score {}".format(params[0], params[1], mae))

    return mae

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize


def plot_gaussian(model):
    plt.figure(1, figsize=(8, 8))

    X_ = np.linspace(0, 25, 1000)
    y_mean, y_std = model.predict(X_[:, np.newaxis], return_std=True)

    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std, alpha=0.5, color='k')

    plt.xlim(0, 25)
    plt.ylim(0, 100)

    plt.title("Posterior\n Log-Likelihood: %.3f"
              % model.log_marginal_likelihood(model.kernel_.theta),
              fontsize=12)

    plt.show()


def expected_improvement(x, gaussian_process, evaluated_loss, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    loss_optimum = np.min(evaluated_loss)

    scaling_factor = -1

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=5, alpha=0.1, epsilon=1e-5):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
        x_list.append(params)
        y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    model = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(),
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        next_sample = sample_next_hyperparameter(expected_improvement, model, yp, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    plot_gaussian(model)

    return xp, yp





# space = np.array([
#     (1.0, 10000.0),
#     (0.1, 100.0)
# ])
# result = bayesian_optimisation(n_iters=100, sample_loss=objective_svr, bounds=space)

space = np.array([
    (0.0001, 25.0)
])
result = bayesian_optimisation(n_iters=50, sample_loss=objective_lasso, bounds=space)

## From: https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
## https://github.com/thuijskens/bayesian-optimization



