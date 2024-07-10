import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                                           # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
print(df.estimated_value_class.value_counts())                     # displays distribution of estimated_value_class
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
log = LogisticRegression()                                         # assigning alias lg to LogisticRegression() function
print(log.fit(X,y2))                                               # training the logistic regression model
print(log.score(X,y2))                                             # test the logistic regression model
X_train, X_test, y2_train, y2_test = train_test_split(X,y2)        # randomly split X,y data to 2 X,y (train,test) sets
print(X_train.shape, y2_train.shape, X_test.shape, y2_test.shape)  # we used default settings 80% is train to 20% test
print(log.fit(X_train, y2_train))                                    # Using 11,250 data points to train model
print(log.score(X_test, y2_test))