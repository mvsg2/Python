import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('E:\Datasets\iris.csv')
print("The first few rows of the original training set are: \n", dataset.head())
X = dataset.iloc[:, [0,1,2,3]]
Y= dataset.iloc[:, 4]

min_max_scaler = MinMaxScaler()
X= min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, shuffle=True, stratify=dataset['species'], random_state=10)
X_cross_val, X_val, y_cross_val, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=15)
print(pd.DataFrame(y_val).value_counts('species'))
print(pd.DataFrame(y_train).value_counts('species'))
print(pd.DataFrame(y_cross_val).value_counts('species'))
print(X_train.shape)
print(X_test.shape)
print(X_cross_val.shape)
print(X_val.shape)

print("The first few rows of the scaled x-training set are: \n", pd.DataFrame(X_train).head())

# Building a kNN Model and making predictions...
knn_classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2, weights='distance') # metric='euclidean' gives the same result
knn_classifier.fit(X_train, y_train)
y_pred_train = knn_classifier.predict(X_train)
y_pred_cross_val = knn_classifier.predict(X_cross_val)
y_pred_test = knn_classifier.predict(X_val)

print(y_pred_train.shape)
print(y_pred_cross_val.shape)
print(y_pred_test.shape)

# Printing the Confusion Matrices and Accuracies for Training, Cross-validation, and Testing respectively...
print(f'The confusion matrix for training set is: \n{confusion_matrix(y_pred=y_pred_train, y_true=y_train)}')
print(f"Training accuracy: {100*accuracy_score(y_true=y_train, y_pred=y_pred_train)}%")
print(f'The confusion matrix for training set is: \n{confusion_matrix(y_pred=y_pred_cross_val, y_true=y_cross_val)}')
print(f"Cross-validation accuracy: {100*accuracy_score(y_true=y_cross_val, y_pred=y_pred_cross_val)}%")
print(f'The confusion matrix for testing set is: \n{confusion_matrix(y_pred=y_pred_test, y_true=y_val)}')
print(f"Testing accuracy: {100*accuracy_score(y_true=y_val, y_pred=y_pred_test)}%")

# Printing the Precisions for Training, Cross-validation, and Testing respectively...
print(f"Training Precision is: {100*precision_score(y_true=y_train, y_pred=y_pred_train, average=None)}")
print(f"Cross-validation Precision is: {100*precision_score(y_true=y_cross_val, y_pred=y_pred_cross_val, average=None)}")
print(f"Testing Precision is: {100*precision_score(y_true=y_val, y_pred=y_pred_test, average=None)}")

# Printing the Recalls for Training, Cross-validation, and Testing respectively...
print(f"Training Recall is: {100*recall_score(y_true=y_train, y_pred=y_pred_train, average=None)}")
print(f"Cross-validation Recall is: {100*recall_score(y_true=y_cross_val, y_pred=y_pred_cross_val, average=None)}")
print(f"Testing Recall is: {100*recall_score(y_true=y_val, y_pred=y_pred_test, average=None)}")

# Printing the F1-scores for Training, Cross-validation, and Testing respectively...
print(f"Training F1-score is: {100*f1_score(y_true=y_train, y_pred=y_pred_train, average=None)}")
print(f"Cross-validation F1-score is: {100*f1_score(y_true=y_cross_val, y_pred=y_pred_cross_val, average=None)}")
print(f"Testing F1-score is: {100*f1_score(y_true=y_val, y_pred=y_pred_test, average=None)}")

# Visualizing our model's performance...
plt.title('Training data confusion matrix')
sns.heatmap(confusion_matrix(knn_classifier.predict(X_train), y_train), cmap = 'Greys_r', annot=True, xticklabels=dataset['species'].unique(), yticklabels=dataset['species'].unique(), cbar=False)
plt.show()

plt.title('Testing data confusion matrix')
sns.heatmap(confusion_matrix(knn_classifier.predict(X_val), y_val), cmap = 'Greys_r', annot=True, xticklabels=dataset['species'].unique(), yticklabels=dataset['species'].unique(), cbar=False)
plt.show()

plt.title('Cross-validation data confusion matrix')
sns.heatmap(confusion_matrix(knn_classifier.predict(X_cross_val), y_cross_val), cmap = 'Greys_r', annot=True, xticklabels=dataset['species'].unique(), yticklabels=dataset['species'].unique(), cbar=False)
plt.show()
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y1 = float(le.fit_transform(y_cross_val))
# plot_confusion_matrix(knn_classifier, knn_classifier.predict(X_cross_val), y1)