import numpy as np


def write_answer_list(answers, filename):
    with open(filename, "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))


def write_answer_to_file(answer, filename):
    """Write the answer to the file.

    Args:
        answer(float).
        filename(str).

    """
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


def mserror(y, y_pred):
    """Calculate mse.

    Args:
        y (float).
        y_pred (float).

    Returns:
        (float): mse.

    """
    return np.mean((y - y_pred)**2)


def linear_prediction(X, w):
    """Evaluate prediction for X with using weights.

    Args:
        X (numpy array).
        w (numpy array).

    Returns:
        (numpy array): predictions.

    """
    return X.dot(w)
