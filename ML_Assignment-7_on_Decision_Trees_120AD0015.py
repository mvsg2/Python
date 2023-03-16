import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_score
import time

data = pd.read_csv("E:\\Datasets\\Balance_scale_dataset\\balance-scale.data")
print('The original data is:')
print(data)

new_cols= ['Class Name', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
data.columns = new_cols
time.sleep(1.8)

X_train, X_test, y_train, y_test= train_test_split(data.drop('Class Name', axis=1), pd.DataFrame(data['Class Name']), random_state=183, test_size=0.30)
classifier= DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('\nTraining the model...\n')

time.sleep(3.9)
print('Accuracy:{}%'.format(100*accuracy_score(y_true=y_test, y_pred=y_pred)))
time.sleep(1)
print('Precision:{}%'.format(100*precision_score(y_true=y_test, y_pred=y_pred, average=None)))
time.sleep(1)
print('Recall:{}%'.format(100*recall_score(y_true=y_test, y_pred=y_pred, average=None)))
time.sleep(1)
print('F1-score:{}%'.format(100*f1_score(y_true=y_test, y_pred=y_pred, average=None)))
time.sleep(2)

plt.title('Heatmap for the Confusion Matrix') 
sns.heatmap(data=confusion_matrix(y_true=y_test, y_pred=y_pred), annot=True, cmap='Greys_r', xticklabels=['L', 'R', 'B'], yticklabels=['L', 'R', 'B'])
plt.show()
time.sleep(1)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('The confusion matrix is: \n', cm)

time.sleep(1.6)
cr = classification_report(y_true=y_test, y_pred=y_pred)
print('The classification report is: \n')
print(cr, end='')