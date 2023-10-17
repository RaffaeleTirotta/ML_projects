def warn(*args,**kargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#take dataset from
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/cancer.csv"

#read dataset in padas as dataframe
df = pd.read_csv(URL)

#show 5 random sample
print(df.sample(5))
#print the dataset shape
print(df.shape)

#plot and count the species
df.diagnosis.value_counts().plot.bar()
plt.show()

#identify target column
target = df["diagnosis"]
#features to classify our data
features = df[["radius_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "symmetry_mean"]]

#create a classifier for the model
classifier = LogisticRegression()
classifier.fit(features, target)

#give the score to the trained classifier
#score to the training data
print("The prediction score is: "+str(classifier.score(features, target)))

#predict a flower species with inserted petal features
print("Insert mean radius:")
x = float(input())
print("Insert mean perimeter")
y = float(input())
print("Insert mean area:")
w = float(input())
print("Insert mean smoothness:")
z = float(input())
print("Insert mean compactness:")
h = float(input())
print("Insert mean concavity:")
j = float(input())
print("Insert mean smoothness:")
t = float(input())

print("The flower is part of "+str(classifier.predict([[x, y, w, z, h, j, t]])[0]))
