import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot

#Import all neccessary model for training
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = r.data
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
df["Attrition"] = label.fit_transform(df.Attrition)

dummy_col = [column for column in df.drop('Attrition', axis=1).columns if df[column].nunique() < 20]
data = pd.get_dummies(df, columns=dummy_col, drop_first=True, dtype='uint8')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

X = data.drop('Attrition', axis=1)
y = data.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
X_std = scaler.transform(X)

models = []
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("LR", LogisticRegression(solver = "liblinear", multi_class = "ovr", penalty='l1')))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma ="auto")))
models.append(("RF", RandomForestClassifier(n_estimators = 100)))

results = []
names = []
print('Model, Accuracy, Standard Deviation')
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = "accuracy")
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
logistic = LogisticRegression(solver = "liblinear", multi_class = "ovr", penalty='l1');
logistic.fit(X_train, y_train);
print ("\n ---Logistic Model---")
logistic_roc_auc = roc_auc_score(y_test, logistic.predict(X_test))
print ("Logistic AUC = %2.2f" % logistic_roc_auc)
print(classification_report(y_test, logistic.predict(X_test)))

# Linear Discriminant
ld = LinearDiscriminantAnalysis()
ld.fit(X_train, y_train);
print ("\n ---Linear Discriminant Analysis---")
ld_roc_auc = roc_auc_score(y_test, ld.predict(X_test))
print ("LDA AUC = %2.2f" % ld_roc_auc)
print(classification_report(y_test, ld.predict(X_test)))

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=10, 
    )
rf.fit(X_train, y_train);
print ("\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
lda_fpr, lda_tpr, lda_thresholds = roc_curve(y_test, ld.predict_proba(X_test)[:,1])

plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logistic_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(lda_fpr, lda_tpr, label='Decision Tree (area = %0.2f)' % ld_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
