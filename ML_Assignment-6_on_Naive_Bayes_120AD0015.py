import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, precision_score
import time

data = pd.read_csv('E:\Datasets\pima_indians_diabetes.csv')
print('The original dataset is: \n', data)

columns = np.array(['No_of_times_Pregnant', 'Plasma_Glucose_Conc.', 'Lower_Blood_Pressure', 'Triceps_Skinfold_Thickness', 'Serum_Insulin', 'BMI', 'DPF', 'Age', 'Diabetic'])
data.columns = columns
# columns = list(data.columns)
time.sleep(2)
print('\n         Number of Zeroes in Each Column')
print('-------------------------------------------------')
for i in columns: 
    print('{:25s}   \t:    {:6s}'.format(i, str(data[data[i] == 0].shape[0])))
print()

time.sleep(1)
for i in columns:
    data[i].plot.density()
    plt.title(f'{i}(Original) Density Plot')
    plt.xlabel(f'{i}')
    plt.show()
time.sleep(1.9)
print("Those were a few density plots for each column of the data.\n")
# data1 = data.copy()
# columns1 = np.array(data1.columns)
# for rows in range(data1.shape[0]):
#     for cols in columns1[1:6]:
#         if (data1[cols][rows] == 0):
#             data1[cols][rows] = np.median(data1[cols])

# The code snippet above also replaces all the 0 values of a column with its corresponding median, regardless if a patient is diabetic or not,
# meaning that it doesn't take into consideration that diabetic patients may have different feature values (for the chosen columns-2, 3, 4, and 5) 
# than non-diabetic patients. So, the medians of diabetic and non-diabetic arrays are taken separately and the zeroes are 
# imputed respectively with the corresponding medians.

imputation_matrix = {}
for i in columns:
    # In the line below, using bitwise AND (i.e. &) gives the correct answer whereas using logical AND (i.e. 'and') throws a KeyError saying that the truth value of a Series is ambiguous
    imputation_matrix.update({i : [data[(data['Diabetic']==0) & (data[i]!=0)][i].median(), data[(data['Diabetic']==1) & (data[i]!=0)][i].median()]})          # update the old dictionary with the imputed values
time.sleep(1)
# print(f"The medians of the Plasma_Glucose_Conc. column for non-diabetic persons is {imputation_matrix['Plasma_Glucose_Conc.'][0]} and diabetic persons is {imputation_matrix['Plasma_Glucose_Conc.'][1]}")
time.sleep(1)
print("  Median values to be imputed for each column's zeroes")
print('---------------------------------------------------------')
print('Column\t\t\t Non-Diabetic   Diabetic')
for i in imputation_matrix:
    print('{:27s}'.format(i), ' {:10s}\t  {:10s}'.format(str(imputation_matrix[i][0]), str(imputation_matrix[i][1])))

for cols in columns[1:6]:
    temp_arr = []
    for i in range(data.shape[0]):
        if ((data['Diabetic'][i]==0) and data[cols][i]==0):
            temp_arr.append(imputation_matrix[cols][0])
        elif ((data['Diabetic'][i] == 1) and data[cols][i] == 0): temp_arr.append(imputation_matrix[cols][1])
        else: temp_arr.append(data[cols][i])
    data[cols] = np.array(temp_arr)
time.sleep(2.12)
print()
print('The modified (median-filled) dataset is: \n', data)
time.sleep(2.31)
for i in columns:
    data[i].plot.density()
    plt.title(f'{i} (Modified) Density Plot')
    plt.xlabel(f'{i}')
    plt.show()
time.sleep(2.21)
# Phew! Now onto actually fitting the model to this modified data...
X_train, X_test, y_train, y_test= train_test_split(data[columns[0:7]], data['Diabetic'], test_size=0.30, random_state=1748, stratify=data['Diabetic']) # 1748 gives the max accuracy
print('X_train shape is: ', X_train.shape)
print('X_test shape is: ', X_test.shape)
print('y_train shape is: ', y_train.shape)
print('y_test shape is: ', y_test.shape)

print("\nTraining the model...\n")
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

time.sleep(3.45)
print(f"The accuracy of the model is: {100*accuracy_score(y_test, y_pred)}%")
print(f"The precision of the model is: {100*precision_score(y_test, y_pred)}%")
print(f"The recall of the model is: {100*recall_score(y_test, y_pred)}%")
print(f"The f1-score of the model is: {100*f1_score(y_test, y_pred)}%"); time.sleep(2.38)
print('\nOur model has misclassified a total of %d examples out of %d in the test-set.\n' %((y_pred!=y_test).sum(), X_test.shape[0]))

time.sleep(2)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(data=cm, cmap='Greys_r', annot=True, linewidth=4.12, linecolor='g')
plt.title("Confusion Matrix for Evaluation of our Model's Performance")
plt.show()
time.sleep(0.92)
print('The confusion matrix is: \n', cm)

time.sleep(2.1)
print('-------------------- CLASSIFICATION REPORT ------------------------------')
cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr, end="")
print('-------------------------------------------------------------------------')