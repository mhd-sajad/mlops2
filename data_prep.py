import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_v1.csv')
X = df.drop("churned",axis=1)
y = df['churned']


X_train,X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Data succesfully prepared")