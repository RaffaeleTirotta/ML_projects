def warn(*args,**kargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#take dataset from
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/iris.csv"

#read dataset in padas as dataframe
df = pd.read_csv(URL)

#show 5 random sample
print(df.sample(5))

#print the dataset shape
print(df.shape)

#plot and count the species
df.Species.value_counts().plot.bar()
plt.show()

#identify target column
target = df["Species"]
#features to classify our data
features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

#create a classifier for the model
classifier = LogisticRegression()
classifier.fit(features, target)

#give the score to the trained classifier
#score to the training data
print("The prediction score is: "+str(classifier.score(features, target)))

#predict a flower species with inserted petal features
print("Insert Sepal Length (Cm):")
x = float(input())
print("Insert Sepal Width (Cm):")
y = float(input())
print("Insert Petal Length (Cm):")
w = float(input())
print("Insert Sepal Width (Cm):")
z = float(input())

print("The flower is part of "+str(classifier.predict([[x, y, w, z]])[0]))
