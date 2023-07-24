import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from sklearn.linear_model import LinearRegression  # Add this import statement
#import statsmodels.api as sm
from statsmodels.formula.api import ols
from scan.functions import cluster_regression
from matplotlib import cm

# Read data from the CSV file

kluster = 15

BB, ww = cluster_regression("rdata.csv", kluster)












data = pd.read_csv("C:/Users/ryang/PycharmProjects/Scan/rdata.csv")

cmap = cm.get_cmap('tab10', kluster)
x=data['x']
y=data['y']
data['ww']=ww

cmap = cm.get_cmap('tab10', kluster)

# Plot the data with colors based on 'ww' values
ww_colors = {ww_val: cmap(i) for i, ww_val in enumerate(sorted(data['ww'].unique()))}

# Plot the data with colors based on 'ww' values
for ww_val, group in data.groupby('ww'):
    plt.scatter(group['x'], group['y'], color=ww_colors[ww_val], label=f'ww = {ww_val}')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot with Colors based on ww')
plt.show()
