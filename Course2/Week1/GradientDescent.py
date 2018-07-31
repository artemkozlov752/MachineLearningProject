import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from utils import write_answer_to_file, mserror, linear_prediction
from config import Config


PATH = Config.get_dataset_path('Week_1', 'advertising')
FEATURE_COLUMNS = Config.get_feature_columns('Week_1', 'advertising')
LABEL_COLUMN = Config.get_label_column('Week_1', 'advertising')
ANSWER_PATH = Config.get_answer_path('Week_1', 'advertising')


def get_feture_matrix_normalized(X):
    """Return normalized X matrix unit column.

    Args:
        X (np.array): feature array.

    """
    means, stds = X.mean(axis=0), X.std(axis=0)
    X = (X - means) / stds
    unit_vector = np.ones(X.shape[0], dtype='float32')[:, np.newaxis]
    X = np.hstack([X, unit_vector])
    return X


def get_train_test(adver_data):
    X = np.array(adver_data[FEATURE_COLUMNS])
    X = get_feture_matrix_normalized(X)
    y = np.array(adver_data[LABEL_COLUMN])
    return X, y


def get_normal_weights(X, y):
    """Return normal weights.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.

    """
    reverse_multiple = np.linalg.inv(X.T.dot(X))
    return reverse_multiple.dot(X.T).dot(y)


def task_1(y):
    """Get task 1 solution.

    Return mse if a(x) = median(y).

    Args:
        X (np.array): feature array.
        y (np.array): label vector.

    """
    return mserror(y, np.median(y))


def task_2(X, y):
    """Get task 2 solution.

    Predict sales with weights found using the normal equation in the case
    of mean investments in advertising on TV, radio and in newspapers.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.

    """
    normal_weights = get_normal_weights(X, y)
    answer2 = normal_weights.dot(X.mean(axis=0))
    return answer2


def task_3(X, y):
    """Get task 3 solution.

    Calculate mse if weights are normal equation solution.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.

    """
    normal_weights = get_normal_weights(X, y)
    preds = linear_prediction(X, normal_weights)
    return mserror(y, preds)


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    """Return gradient step coefficients.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.
        w: (np.array): weights vector.
        train_ind (int): random index.
        eta (float): step coefficient.

    Returns:
        (np.array): new step weights.

    """
    y_pred = linear_prediction(X[train_ind, :], w)
    grad0 = y_pred - y[train_ind]
    grad1 = X[train_ind, 0] * grad0
    grad2 = X[train_ind, 1] * grad0
    grad3 = X[train_ind, 2] * grad0
    return w - 2 * eta/y.shape[0] * np.array([grad1, grad2, grad3, grad0])


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    """Realizes stochastic gradient algorithm.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.
        w_init: (np.array): weights vector initialized.
        eta (float): step coefficient.
        max_iter (float): maximum iteration amount.
        min_weight_dist (float): stop algorithm distant between weights.
        seed (int): random seed.
        verbose (bool): print flag.

    Returns:
        (np.array): algorithm's weights.
        (list): algorithm's errors.

    """
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)
    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        w_prev = w
        w = stochastic_gradient_step(X, y, w_prev, random_ind, eta)
        y_pred = linear_prediction(X, w)
        errors.append(mserror(y, y_pred))
        weight_dist = np.linalg.norm(w - w_prev)
        iter_num += 1
    return w, errors


def task_4_grad_desc(X, y):
    """Run stochastic gradient descent.

    Uses the following parameters:
    max_iter = 10^5 iterations;
    w_init = 0;
    eta = 0.01;
    seed = 42.

    Args:
        X (np.array): feature array.
        y (np.array): label vector.

    Returns:
        (np.array): new step weights.

    """
    w_init = np.zeros(4)
    stoch_grad_weights, stoch_errors = stochastic_gradient_descent(
        X, y, w_init, eta=1e-2, max_iter=1e5,
        min_weight_dist=1e-8, seed=42, verbose=False
    )
    return stoch_grad_weights, stoch_errors


def task_4_plot(range_plot, stoch_errors):
    """Plot first 50 iterations errors.

    Args:
        (list): algorithm's errors.

    """
    plt.figure()
    plt.plot(range_plot, stoch_errors)
    plt.xlabel('Iteration number')
    plt.ylabel('MSE')


def task_4(X, y):
    """Realizes stochastic gradient realization."""
    stoch_grad_weights, stoch_errors = task_4_grad_desc(X, y)
    task_4_plot(range(50), stoch_errors[:50])
    task_4_plot(range(len(stoch_errors)), stoch_errors)
    answer4 = stoch_errors[-1]
    return answer4


def main():
    adver_data = pd.read_csv(PATH)
    X, y = get_train_test(adver_data)
    write_answer_to_file(task_1(y), os.path.join(ANSWER_PATH, 'task_1.txt'))
    write_answer_to_file(task_2(X, y), os.path.join(ANSWER_PATH, 'task_2.txt'))
    write_answer_to_file(task_3(X, y), os.path.join(ANSWER_PATH, 'task_3.txt'))
    write_answer_to_file(task_4(X, y), os.path.join(ANSWER_PATH, 'task_4.txt'))
