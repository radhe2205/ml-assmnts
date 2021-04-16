import pandas as pd
import numpy as np

# Please place the data files beside this file to run.
# ---------------------------
# Total squared error:      |
# a) 5.7154872421368434     |
# b) 34559429888.128456     |
# ---------------------------

# Results Investigation:

# In the first part we get good approximation.

# In the second part the results are too bad. Even predicting 0 value for y yields better results than the closed form
# solution. In the second part the weights calculated are around 100-1000 in value. Which is very high.
# It makes sense, as the solution tries to predict from 500 data points, while making prediction for 501 values(Including bias).
# Since to solve for 501 values we need at least 501 data points, our solution outputs one random set of weights which satisfy the equation.

# For example, if we are given only one point on a 2D plane, we can pass infinite number of lines through that point. We need at least
# 2 points to estimate the relationship between x and y. Similarly in 3D, if we are given only 2 points, there are infinite number
# of planes passing through it. Our solution will output any random plane which the line passes through. That is why when we test
# our weights on the test data, the random plane outputs huge error. As the solution is random.

# Another thing noticed in the 2nd part is that the training error is zero. That is, when we try to predict training data,
# the prediction is exact. While this is not true for 1st part. This is because in the first part, we have calculated
# approximate weight, while in the second part, we have found the exact multi-dimenstional plane which passes through all
# the points.

# When all 1000 points are used for training, in the second part, the results are better.

def get_sse_error(data_file, resp_file):
    x = pd.read_csv(data_file, delim_whitespace=True, header=None)
    y = pd.read_csv(resp_file, delim_whitespace=True, header=None)

    x[len(x.columns)] = np.ones(len(x)) # Adding 1 in the front for the bias

    x_train = x[:int(len(x)/2)]
    y_train = y[:int(len(y)/2)]
    x_test = x[int(len(x)/2):]
    y_test = y[int(len(y)/2):]

    # w_hat = inv(Xt * X) * Xt * y -> Closed form solution for weights
    w_hat = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    y_pred = x_test.dot(w_hat)
    error = y_test[y_test.columns[0]]-y_pred[y_pred.columns[0]]

    sse = error.T.dot(error)

    return sse

print(get_sse_error("pred1.dat", "resp1.dat"))
print(get_sse_error("pred2.dat", "resp2.dat"))
