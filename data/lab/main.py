import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt


def calculate_mae(y_true, y_pred):
    n = len(y_true)
    mae = 0

    for i in range(n):
        mae += abs(y_true[i] - y_pred[i])

    mae = mae / n
    return mae

# dataframe import
df = pd.read_csv("data\\lab\\data.csv")

# data normalization
columns_to_normalize = ['Population', 'GDP ($ per capita)', 'Literacy (%)', 'Development Index']

scaler = StandardScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# save normilized data into CSV
df.to_csv('data\\lab\\normalized_data.csv', index=False)

# chart of normilized data
sns.set(style="whitegrid")

# span diagram
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[columns_to_normalize])
plt.title('Диаграмма размаха нормализованных данных')
plt.xlabel('Параметры')
plt.ylabel('Значения')
plt.xticks(rotation=45)
# plt.show()

# dev index import
di = pd.read_csv("data\\lab\\data.csv")

# histogram of Development index
plt.figure(figsize=(10, 6))
plt.hist(di['Development Index'], bins=30, edgecolor='k')
plt.title('Гистограмма исходных данных для Development Index')
plt.xlabel('Development Index')
plt.ylabel('Частота')
plt.grid(True)
# plt.show()

# normilized dataframe import
df = pd.read_csv("data\\lab\\normalized_data.csv")

# Division dataframe by signs (X) and target variable (y)
X = df.drop('Development Index', axis=1)
y = di['Development Index']

# devide data frame on test and training sets 1:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# model initialization
ada_model = AdaBoostRegressor()

# train model on train set
ada_model.fit(X_train, y_train)

# Prediciton of dev index wiht test data frame
y_pred = ada_model.predict(X_test)

# denormilization of predicted values
mean = df['Development Index'].mean()
std = df['Development Index'].std()
y_pred_denorm = y_pred * std + mean

# round if needed
y_pred_rounded = y_pred_denorm
y_test_rounded = y_test

print("predicted denorm", len(y_pred_denorm))
print(y_pred_rounded)

print("test data ", len(y_test_rounded))
print(y_test_rounded)

comparison_array = pd.DataFrame({'Original': y_test_rounded, 'Predicted': y_pred_denorm}).reset_index(drop=True)

# bar chart of real and predicted values
real = comparison_array['Original']
pred = comparison_array['Predicted']

comparison_array['Predicted']=comparison_array['Predicted'].apply(int)
test_plot = comparison_array
print(comparison_array.values)

test_plot.plot(kind="bar")
plt.rcParams['figure.figsize']=[20,2]
plt.xlabel("Samples")
plt.ylabel("Development index")
plt.show()

# calculate MAE
mae = calculate_mae(real, pred)
print("MAE:", mae)


#//////////////////////////////////////////////////////////////////////////////////////////////////
# confusion matrix
y_true = round(comparison_array['Original'])
print(y_true)
y_pred = round(comparison_array['Predicted'])

a_true = y_true.values
a_pred = y_pred.values

# classes
classes = np.unique(np.concatenate([a_true, a_pred]))
cm = confusion_matrix(a_true, a_pred, labels=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('original')
plt.title('Confusion matrix')
plt.show()

