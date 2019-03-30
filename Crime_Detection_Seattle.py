#!/usr/bin/env python
# coding: utf-8

# ## Initial Cleaning of Data
# 
# Using Excel's excellent data processing tools, I transformed the data as follows:
# 1. Split the data in the Report Location column, by using the comma as the delimiter.
# 2. Cleaned up occurances of closing and opening parentheses.
# 
# Both these operations were done using Excel's data to text to column options. 
# 
# The new columns are named rLatitude and rLongitude




import pandas as pd
import numpy as np

from sklearn import preprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import imread


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ### Reading in data




df = pd.read_excel('data.xlsx')


# ### Exploratory Data Analysis




df.head()





df.describe()





df.Type.unique()


# ### Preprocessing data in order to obtain histogram
# 
# The labels are stored in the Type column. While the values make sense to us humans, our machine learning algorithms won't be able to make sense of these strings. It is for this purpose that we convert these string values to specific numeric labels. I decided to go with a label encoder over one hot encoding as I see a need for the data to still remain nominal.




label_encoder = preprocessing.LabelEncoder()
df['Type'] = label_encoder.fit_transform(df.Type)




df.Type.unique()





plt.hist(df.Type, histtype='bar', bins=4, color='gray')




label_encoder.inverse_transform(df.Type.unique())


# ##### **The most common incident for which 911 was called is because of Beaver Attacks, as shown in the histogram above.**

# ## 2




a = df.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.5, c= 'Type', cmap=plt.get_cmap('Dark2'), figsize=(10, 8))

img = plt.imread('./seattle.PNG')
plt.imshow(img, zorder=0, extent=[-122.469940, -122.140100, 47.500200, 47.732000])
plt.show()





a = df.plot(kind='scatter', x='rLongitude', y='rLatitude', alpha=0.5, c= 'Type', cmap=plt.get_cmap('Dark2'), figsize=(10, 8))

img = plt.imread('./seattle.PNG')
plt.imshow(img, zorder=0, extent=[-122.469940, -122.140100, 47.500200, 47.732000])
plt.show()


# #### 2B
# 
# As we can see from the first plot, there appear to be a few points that seem like outliers. These are visible by the contrasting colors. That is the brown spot among the sea of light blue or the few brown spots among the darker blue spots. These are data points that are worthy of further exploration.
# 
# 
# The second plot seems to be too clean to be true in some places, while in others, it doesn't seem as though they are all within the city. Some of the yellow points, based on my simple extrapolation and napkin analysis seem to be out over the water. 

# # 3
# 
# 3A and 3C are written below the algorithms. 3B is answered collectively at the end of this section. 



from sklearn.model_selection import train_test_split

indices = np.array(list(range(len(df.Type))))

# Train and test indices split
train_indices, test_indices = train_test_split(indices, test_size=0.18, shuffle=True, random_state=9)




def get_train_test(dataFrame, indices, col):
    data = dataFrame.loc[indices]
    val = data[col].values
    x = val[:, 1:]
    y = val[:, 0]
    return x, y





train_x, train_y = get_train_test(df, train_indices, ['Type', 'Latitude', 'Longitude'])
test_x, test_y = get_train_test(df, test_indices, ['Type', 'Latitude', 'Longitude'])


# ## Logistic Regression using 2 attributes




from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=9, max_iter=180, C=5.4)
reg.fit(train_x, train_y)
pred = reg.predict(test_x)
print('Number of correct predictions : ', end='')
tot_correct = np.sum(pred == test_y)
print(np.sum(pred == test_y))
print('Total Accuracy : {} % '.format(100 * tot_correct / len(pred)))


# 
# By using only latitude and longitude, we are able to obtain a fairly good accuracy in predicting the values. One caveat is that we need the labels during training, at the very least. <br>
# In the above method, we used multi-class Logistic Regression. It used newton-cg method to solve for the pseudoinverses which intern allows us to obtain the parameters. 

# ## Logistic Regression using 4 attributes



train_x, train_y = get_train_test(df, train_indices, ['Type', 'Latitude', 'Longitude', 'rLatitude', 'rLongitude'])
test_x, test_y = get_train_test(df, test_indices, ['Type', 'Latitude', 'Longitude', 'rLatitude', 'rLongitude'])

reg = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=9, max_iter=180, C=5.4)
reg.fit(train_x, train_y)
pred = reg.predict(test_x)
print('Number of correct predictions : ', end='')
tot_correct = np.sum(pred == test_y)
print(np.sum(pred == test_y))
print('Total Accuracy : {} % '.format(100 * tot_correct / len(pred)))


# By using the reported latitude and longitude, we obtain a minimal improvement in accuracy and correct predictions. Thus, it is good enough if we use the reduced feature set. 

# ## Logistic Regression using 2 attributes and Stochastic Gradient Descent 




train_x, train_y = get_train_test(df, train_indices, ['Type', 'rLatitude', 'rLongitude'])
test_x, test_y = get_train_test(df, test_indices, ['Type', 'rLatitude', 'rLongitude'])

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(alpha=0.009, random_state=9, eta0=0.01, max_iter=108)
classifier.fit(train_x, train_y)
clf_pred = classifier.predict(test_x)
print('Number of correct predictions : ', end='')
tot_correct = np.sum(pred == test_y)
print(np.sum(pred == test_y))
print('Total Accuracy : {} % '.format(100 * tot_correct / len(pred)))


# While I expected SGD(Stochastic Gradient Descent) to provide a much better improvement, we obtain just a little improvement when compared to the newton method. While I would prefer newton in this scenario, if there were more data, then, SGD would be the preferred method of obtaining the parameters which correspond to the minima of the loss function. 




train_x, train_y = get_train_test(df, train_indices, ['Type', 'Latitude', 'Longitude'])
test_x, test_y = get_train_test(df, test_indices, ['Type', 'Latitude', 'Longitude'])

from sklearn.cluster import KMeans

clustering = KMeans(n_clusters=4, random_state=9, init='k-means++', max_iter=900, precompute_distances=True)
clustering.fit(train_x, train_y)
prediction = clustering.predict(test_x)

print('Number of correct predictions : ', end='')
print(np.sum(clf_pred == test_y))


# My first thought in solving this problem was in using a clustering method. I tried using DBSCAN, however, the number of correct results was just 9 out of 273. Using kMeans, I obtained a better scores, albeit it is still not as good as the other methods.

# ### 3B
# 
# While the logistic models are not concerned if the underlying data is Euclidean or otherwise, the clustering method does take into consideration the distance. Considering that the data points are spread out over a single city, we need not worry about the curvature of the earth causing issues when we use Euclidean distance. 

# ### 3E
# 
# The insight that I obtained from this dataset is that while the most common reason to call 911 is beaver attacks, these attacks don't quite happen all over the city. This is quite interesting as we can recommend safety practices to these counties. Further, we can identify places where such things are very rare. Folks who have a phobia of beavers, might be interested in buying houses there. Similarly, we can draw insights on the other types of crime. 
# 
# By overlaying data over the map, we can see that certain regions of Seattle are safer as there appear to be no crime there. 
