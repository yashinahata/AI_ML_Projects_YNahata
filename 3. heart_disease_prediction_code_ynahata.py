"""
Machine Learning for Heart Disease Prediction based on various lifestyle, clinical, and demographic factors
Author: Yashi Nahata
Dataset: University of California, Irvine (UCI) Heart Disease Dataset
Model deployed: K-means clustering
References: Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. Heart Disease. 
            UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
"""

# 1. Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Reading and exploring the UCI Heart Disease data
dataset = pd.read_csv('C:/Users/yasha/project_1/heart.csv')
dataset.info()
dataset.shape
dataset.head()

# 3. Transforming categorical variables and normalizing clinical variables
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale]=standardScaler.fit_transform(dataset[columns_to_scale])
dataset.shape
dataset.head()

# 4. Splitting the transformed data from previous step into training and testing data
y = dataset['target']
x = dataset.drop(['target'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# 5. Identifying optimal model input parameters for best accuracy
knn_scores = []
for k in range (1,15):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test.values)
    knn_scores.append(accuracy_score(y_test, y_pred))
print(knn_scores)
print('Best choice of k:', np.argmax(knn_scores)+1)
plt.plot(range(1,15), knn_scores)
plt.xlabel("No. of clusters")
plt.ylabel("Model accuracy %")
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in current_values]) 
plt.show()

# 6. Running the model and analyzing its accuracy and results
k = np.argmax(knn_scores) + 1
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_classifier.fit(x_train, y_train)
y_pred = knn_classifier.predict(x_test.values)
accuracy = np.sum(y_pred==y_test)*100/len(y_test)
print('Prediction accuracy:',round(accuracy), '%')
unique, counts = np.unique(y_pred, return_counts = True)
result = np.vstack((unique, counts)).T
print(result)
centers = knn_classifier.cluster_centers_
probability = knn_classifier.predict_proba(x_test.values)
print(probability)
print(y_pred)
print(y_test.values)
