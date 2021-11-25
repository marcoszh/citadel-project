# import numpy as np
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score

# import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the provided data set for Credit Card dataset
data = pd.read_csv("creditcard.csv")
# print(data.head(5))
# print(data.describe())
# print(data.shape)

data_count_class = data
class_count = pd.value_counts(data['Class'], sort=True).sort_index()

fraud_data = data[data_count_class.Class == 1]
normal_data = data[data_count_class.Class == 0]

# print(fraud_data.shape)
# print(normal_data.shape)

# We need to Scale the Amount as it is in different Scale
model_data = data.drop(['Time'], axis=1)
model_data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
print(model_data.head())

# split into train and test
model_train = model_data.drop("Class", 1).values
# model_train.shape
model_test = model_data["Class"].values

# sampling cuz imbalanced data
sampling_train = model_train
sampling_test = model_test
sampler = SMOTE(random_state=0, n_jobs=-1)
model_train_lr, model_test_lr = sampler.fit_sample(sampling_train, sampling_test)

# split
X_train, X_test, Y_train, Y_test = train_test_split(model_train_lr, model_test_lr, test_size=0.25, random_state=0)

# RandomForest_model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0, n_jobs=-1,
#                                             verbose=True)
# RandomForest_model.fit(X_train, Y_train)
# RandomForest_predict = RandomForest_model.predict(X_test)

start_t = time.time()

lr = LogisticRegression(max_iter=300, random_state=0, n_jobs=-1, verbose=True)
lr.fit(X_train, Y_train)

print("time elapsed: {:3f}".format(time.time()-start_t))
# print(lr.get_params())
print(lr.coef_)
print(lr.intercept_[:,None])
lr_prediction = lr.predict(X_test)
# print(lr_prediction)

sc_lr_accuracy = accuracy_score(Y_test, lr_prediction)
sc_lr_recall = recall_score(Y_test, lr_prediction)
sc_lr_cm = confusion_matrix(Y_test, lr_prediction)
sc_lr_auc = roc_auc_score(Y_test, lr_prediction)

print("Model has a Score_Accuracy: {:.3%}".format(sc_lr_accuracy))
print("Model has a Score_Recall: {:.3%}".format(sc_lr_recall))
print("Model has a Score ROC AUC: {:.3%}".format(sc_lr_auc))

# sc_rf_accuracy = accuracy_score(Y_test, RandomForest_predict)
# sc_rf_recall = recall_score(Y_test, RandomForest_predict)
# sc_rf_cm = confusion_matrix(Y_test, RandomForest_predict)
# sc_rf_auc = roc_auc_score(Y_test, RandomForest_predict)

# print("Model has a Score Accuracy: {:.3%}".format(sc_rf_accuracy))
# print("Model has a Score Recall: {:.3%}".format(sc_rf_recall))
# print("Model has a Score ROC AUC: {:.3%}".format(sc_rf_auc))
