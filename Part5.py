import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
X1 = X                                               # duplicate of X
y = df.estimated_value                               # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
X_train, X_test, y_train, y_test = train_test_split(X,y)        # randomly split X,y data to 2 X,y (train,test) sets
X1_train, X1_test, y2_train, y2_test = train_test_split(X1,y2)        # randomly split X,y data to 2 X,y (train,test) sets
knn_reg = KNeighborsRegressor()                          # assign knn_reg to the KNeighborRegressor() function
knn_class = KNeighborsClassifier()                       # assign knn_class to the KNeighborClassifier function
knn_reg.fit(X_train, y_train)                            # fit a knn_reg model using 11,250 data points
print('knn_reg score = ', knn_reg.score(X_test, y_test)) # score the knn_reg model based on 11,250 data points using the 3,750 data points
knn_class.fit(X1_train, y2_train)                        # fit a knn_class model using 11,250 data points
print('knn_class score = ', knn_class.score(X1_test, y2_test)) # score the knn_class model base on 11,250 data points using the 3,750 data points
y2_pred = knn_class.predict(X_test)                      # assigns y2_pred to knn_class prediction of X_test data
print(confusion_matrix(y2_test, y2_pred))                # displays correct on diagonal, typeI right, typeII left, all 3,750