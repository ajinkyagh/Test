import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                               # predictor variable lower case y as array
sns.boxplot(X['lotSize'])
plt.show()
sns.boxplot(X['yearBuilt'])
plt.show()
sns.boxplot(X['priorSaleAmount'])
plt.show()
