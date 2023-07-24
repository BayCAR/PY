import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#from sklearn.linear_model import LinearRegression  # Add this import statement
#import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import cm

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
kluster = 12
sdd = np.full(kluster, 1000)
pii = np.full(kluster, 1)
pii0 = pii * 9
n_n = data.shape[0]
inc = np.empty(n_n)

pvar = np.diag([1000, 1000, 1000])

cbeta = np.percentile(y, np.linspace(0, 100, kluster))
pBeta = np.vstack((cbeta, np.linspace(0, 1e-6, kluster), np.linspace(0, 1e-10, kluster)))
Beta = pBeta
T = np.zeros((n_n, kluster))

for i in range(30):
    for j in range(kluster):
        T[:, j] = pii[j] * norm.pdf(y - np.dot(x[['intercept', 'x', 'x2']], Beta[:, j]), 0, sdd[j])

    row_sums = T.sum(axis=1)
    T = T / row_sums[:, np.newaxis]
    pii = np.sum(T, axis=0) / n_n
    if len(pii) == len(pii0):
        if np.sum(pii - pii0) == 0 and i > 3:
            break

    pii0 = pii
    kluster0 = kluster
    ww = np.empty(n_n)
    max_column_indices = np.argmax(T, axis=1)
    ww[:] = max_column_indices
    tbww = pd.Series(ww).value_counts()
    print(tbww)
    grps = tbww.index.astype(int)
    kluster = len(grps)
    pii = pii[grps]
    Beta = Beta[:, grps]
    T = T[:, grps]
    print(i)
    #for j in range(kluster):
    for j in range(kluster - 1, -1, -1):
        for w in range(n_n):
            inc[w] = T[w, j] == np.max(T[w, :])
            #vc=pd.Series(inc)
        inc_array = np.array(inc)  # Convert inc to a NumPy array
        x_inc = x[inc == 1]
        x_inc = x_inc[['intercept', 'x', 'x2']]
        y_inc = y[inc == 1]
        x_inc_T = x_inc.T
        #if i<=5:
            #Beta[:, j]  = np.linalg.solve(pvar + np.dot(x_inc_T, x_inc), np.dot(pvar, pBeta[:, j]) + np.dot(x_inc_T,
        # y_inc))
        Beta[:, j] = np.dot(np.linalg.inv(np.linalg.inv(pvar) + np.dot(x_inc_T, x_inc)), (np.dot(np.linalg.inv(pvar),
                                                                                               pBeta[:, j]) + np.dot(x_inc_T, y_inc)))
        pBeta=Beta
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



cmap = cm.get_cmap('tab10', kluster)
print(xx)
# Plot the data with colors based on ww values
ww_colors = {ww_val: cmap(i) for i, ww_val in enumerate(sorted(xx['ww'].unique()))}

# Plot the data with colors based on ww values
for ww_val, group in xx.groupby('ww'):
    plt.scatter(group['x'], group['y'], color=ww_colors[ww_val], label=f'ww = {ww_val}')

plt.legend()
plt.show()

plt.plot(x, y, 'o', label='Original Data')

# Loop over each j in 1 to kluster
for j in range(1, kluster + 1):
    # Calculate the corresponding y values for the j-th cluster using the Betas
    y_values = Beta[0, j - 1] + Beta[1, j - 1] * x + Beta[2, j - 1] * data['x2']

    plt.plot(x, y_values, 'o', markersize=6, label=f'Cluster {j}')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()