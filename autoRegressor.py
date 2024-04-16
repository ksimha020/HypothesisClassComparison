#designed by Kshitij Simha R (2022A7PS0572G)
# Importing all necessary libraries.
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def batchCreation(numDependancies, dataset):
    x_data, y_data = [], []
    for i in range(numDependancies, len(dataset)-numDependancies):
        x_data.append(dataset[i-numDependancies:i])
        y_data.append(dataset[i])
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data

def train_test_split(trainPercent, x_data, y_data):
    li_test_train = np.array(([1]*int(trainPercent*len(x_data))+[0]*(len(x_data)-int(trainPercent*len(x_data)))))
    #np.random.shuffle(li_test_train)
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(li_test_train)):
        if li_test_train[i]:
            x_train.append(x_data[i])
            y_train.append(y_data[i])
        else:
            x_test.append(x_data[i])
            y_test.append(y_data[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def lossCalc(numDependancies, dataset, train_percent, confirmedDependancy=False):
    x_data, y_data = batchCreation(numDependancies, dataset)
    x_train, y_train, x_test, y_test = train_test_split(train_percent, x_data, y_data)
    model = LinearRegression()
    model.fit(x_train, y_train)
    if confirmedDependancy:
        return model, x_data, y_data
    loss = np.sum((model.predict(x_test)-y_test)**2)
    return loss

train_percent = 0.7
df = pd.read_csv('Data/META.csv')
dataset = np.array(df['Close'])
end_dependacy = int((1-train_percent)*len(dataset))
loss = float('inf')
numDepedancies = None

for i in range(10, end_dependacy):
    lossCompare = lossCalc(i, dataset, train_percent)
    if loss > lossCompare:
        print("Loss improved, dependancies changed from", numDepedancies, "to", i)
        loss = lossCompare
        numDepedancies = i
print(f"Final answer contains Linear Model needing previous {numDepedancies} datapoints")

model, x_data, y_data = lossCalc(numDepedancies, dataset, train_percent, True)
y_pred = np.array(model.predict(x_data))

plt.figure()
plt.plot(y_data, label='true')
plt.plot(y_pred, label='predicted')
plt.legend()
plt.title('Linear Model vs Test')
plt.show()