import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
X1 = X
pca = PCA(4)
X_transformed = pca.fit_transform(X)
y = df.estimated_value                               # predictor variable lower case y as array
y1 = df.estimated_value
lg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y)        # randomly split X,y data to 2 X,y (train,test) sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)        # randomly split X,y data to 2 X,y (train,test) sets
print(X_transformed.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # we used default settings 80% is train to 20% test
lg.fit(X_train, y_train)                                         # Using 11,250 data points to train model
print('PCA = ', lg.score(X_test, y_test))                        # Using  3,750 data points to test/evaluate R2 of model
lg.fit(X1_train,y1_train)                                        # Using 11,250 data points to train model
print('non-PCA = ', lg.score(X1_test, y1_test))                  # Using  3,750 data points to test/evaluate R2 of model