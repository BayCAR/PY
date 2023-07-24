import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression  # Add this import statement
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Read data from the CSV file
data = pd.read_csv("C:/Users/ryang/PycharmProjects/Scan/data.csv")

# Rename columns
data["y"] = data["umean"]
data = data[(data["col"] < 350) & (data["col"] > 200)]
data = data[['x', 'y']]
data["intercept"] = 1
data['x2'] = data['x'] ** 2
y = 5000 - data["y"]

# Add a constant column (intercept) to the x dataframe
x = data[["intercept", "x", "x2"]]

# Define the 'kluster' variable before its first usage
kluster = 10
sdd = np.full(kluster, 1000)
pii = np.full(kluster, 1)
pii0 = pii * 9
n_n = data.shape[0]
inc = np.full(n_n, np.nan)

pvar = np.diag([100, 100, 100])

cbeta = np.linspace(min(y), max(y), kluster)
pBeta = np.vstack((cbeta, np.linspace(0, -0.1, kluster), np.linspace(0, -0.001, kluster)))
Beta = pBeta
T = np.zeros((n_n, kluster))

for i in range(100):
    for j in range(kluster):
        T[:, j] = pii[j] * norm.pdf(y - np.dot(x[['intercept', 'x', 'x2']], Beta[:, j]), 0, sdd[j])

    row_sums = T.sum(axis=1)
    # Replace each row with the proportion of that row
    T = T / row_sums[:, np.newaxis]
    pii = np.sum(T, axis=0) / n_n

    if len(pii) == len(pii0):
        if np.sum(pii - pii0) == 0 and i > 3:
            break

    pii0 = pii
    kluster0 = kluster
    ww = np.empty(n_n)

    max_column_indices = np.argmax(T, axis=1)
    # Update vector w with the column indices
    ww[:] = max_column_indices
    tbww = pd.Series(ww).value_counts()
    grps = tbww.index.astype(int)
    kluster = len(grps)
    pii = pii[grps]
    Beta = Beta[:, grps]
    T = T[:, grps]
    print(i)
    for j in range(kluster):
        inc = np.empty(n_n)
        for w in range(n_n):
            inc[w] = T[w, j] == np.max(T[w, :])
            vc=pd.Series(inc)

        inc_array = np.array(inc)  # Convert inc to a NumPy array
        x_inc = x[inc == 1]
        y_inc = y[inc == 1]
        x_inc_T = x_inc.T
        if i<=5:
            Beta[:, j]  = np.linalg.solve(pvar + np.dot(x_inc_T, x_inc), np.dot(pvar, pBeta[:, j]) + np.dot(x_inc_T, y_inc))

        x_inc = x_inc[['intercept', 'x', 'x2']]
        sdd[j] = np.apply_along_axis(np.std, 0, y_inc - np.dot(x_inc, Beta[:, j]))

        if i > 5:
            xx = pd.DataFrame(x)
            xx['y'] = y
            xx['ww'] = pd.Categorical(ww)
            # Create the linear regression model
            lm = ols('y ~ x + x2 + C(ww)', data=xx).fit()
            sm=lm.summary()
            coef = lm.params

            Beta[0, 0] = coef['Intercept']
            Beta[1, :] = coef['x']
            Beta[2, :] = coef['x2']
            Beta[0, 1:] =[coef[0] + m for m in coef[1:(len(coef)-2)]]

    mean_sdd = np.mean(sdd)
    # Repeat the mean value 'kluster' times to create the new 'sdd' array
    sdd = np.full(kluster, mean_sdd)

# Plot the clustered data


plt.subplot(1, 2, 1)
plt.plot(data['x'], data['y'])
for j in range(1, kluster + 1):
    plt.plot(data['x'], Beta[1, j - 1] + Beta[2, j - 1] * data['x'] + Beta[3, j - 1] * data['x2'], 'o', markersize=0.6,
             color=j + 1)
T_max = np.zeros(n_n)
ccol = np.zeros(n_n)
for w in range(n_n):
    ccol[w] = np.argmax(T[w, :]) + 1
    T_max[w] = np.max(T[w, :])
plt.subplot(1, 2, 2)
plt.plot(data['x'], data['y'], color=ccol)
plt.show()
