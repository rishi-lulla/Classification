# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

# load the data from the file
data = load_data(r"C:\Users\RISHI\Desktop\Python\KNN\interview.csv", None)
print(data.head())

# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]
print("X = Feature values, all the columns except the last column")
print(X.head())

# y = target values, last column of the data frame
y = data.iloc[:, -1]
print("y = target values, last column of the data frame")
print(y.head())

# filter out the applicants that got selected
admitted = data.loc[y == 1]
print("applicants that got selected")
print(admitted.head())

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]
print("applicants that din't get selected")
print(not_admitted.head())

# plots the data points
plt.figure()
plt.scatter(admitted.iloc[:, 0],
            admitted.iloc[:, 1], s=10,
            label='Selected')
plt.scatter(not_admitted.iloc[:, 0],
            not_admitted.iloc[:, 1], s=10,
            label='Not Selected')
plt.grid()
plt.legend()
plt.show()

# we split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# import logistic regression
from sklearn.linear_model import LogisticRegression
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# bulding the LogR model
model_lr = LogisticRegression(solver='lbfgs')
model_lr.fit(X_train, Y_train)

# bulding the NB model
model_nb = GaussianNB()
model_nb.fit(X_train, Y_train)

print("Coefficients of the Logistic regression model")
coef = model_lr.coef_
intercept = model_lr.intercept_
print("Coef: ", coef)
print("Intercept: ", intercept)

model_knn = KNeighborsClassifier(n_neighbors=9) #, p=2, metric='minkowski'
model_knn.fit(X_train, Y_train)



# predicting train set to calculate acuracy of LR model
predicted_classes_lr = model_lr.predict(X_train)

# predicting train set to calculate acuracy of NB model
predicted_classes_nb = model_nb.predict(X_train)

predicted_classes_knn = model_knn.predict(X_train)

print("Confusion Matrix for LR model::")
conf_mat_lr = confusion_matrix(Y_train.tolist(),predicted_classes_lr)
print(conf_mat_lr)
sns.heatmap(conf_mat_lr,annot = True) #ann = {"ha": 'center',"va": 'center'}
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

print("Confusion Matrix for NB model::")
conf_mat_nb = confusion_matrix(Y_train.tolist(),predicted_classes_nb)
print(conf_mat_nb)
sns.heatmap(conf_mat_nb,annot = True) #ann = {"ha": 'center',"va": 'center'}
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

print("Confusion Matrix for Knn model::")
conf_mat_knn = confusion_matrix(Y_train.tolist(),predicted_classes_knn)
print(conf_mat_knn)
sns.heatmap(conf_mat_knn,annot = True, xticklabels = ["Not Selected", "Selected"], yticklabels = ["Not Selected", "Selected"])
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

from matplotlib.colors import ListedColormap
def decision_boundary(x,y,model, title):
    cmap_light = ListedColormap(['cyan','cornflowerblue']) #ate color maps
    cmap_bold = ListedColormap(['darkorange','darkblue'])
    h = 0.1
    x_min,x_max = x.iloc[:,0].min() - 1,x.iloc[:,0].max() +1  # plot the decision boundary. for that, we will assign a color to each
            #3 point in te mesh[x_min,x_max]x[y_min,y_max]
    y_min,y_max = x.iloc[:,1].min() - 1,x.iloc[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.figure() ## put the result into a color plot
    plt.pcolormesh(xx,yy,z,cmap = cmap_light)
    plt.scatter(x.iloc[:,0],x.iloc[:,1],c=y,cmap = cmap_bold,edgecolor = "k",s = 20)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title(title)
    plt.show()
    
### Decision boundary of logisic Reg classifier
decision_boundary(X,y,model_lr, "Decision boundary using LR")
decision_boundary(X,y,model_nb, "Decision boundary using NB")
decision_boundary(X,y,model_knn, "Decision boundary using knn")

#ROC and AUC
from sklearn.metrics import roc_auc_score, roc_curve
plt.figure()

logit_roc_auc = roc_auc_score(Y_test, model_lr.predict_proba(X_test)[:,1])
fpr, tpr, threshold = roc_curve(Y_test, model_lr.predict_proba(X_test)[:,1], 
                                drop_intermediate = False)
plt.plot(fpr, tpr, 'b--', label = 'Logistic Regression (area = %0.3f)' % logit_roc_auc)
nb_roc_auc = roc_auc_score(Y_test, model_nb.predict_proba(X_test)[:, 1])
fpr_nb, tpr_nb, threshold_nb = roc_curve(Y_test, model_nb.predict_proba(X_test)[:,1],
                                         drop_intermediate=False)

plt.plot(fpr_nb, tpr_nb, 'r:', label = 'Naive Bayes (area = %0.3f)' % nb_roc_auc)
knn_roc_auc = roc_auc_score(Y_test, model_knn.predict_proba(X_test)[:, 1])
fpr_knn, tpr_knn, threshold_knn = roc_curve(Y_test, model_knn.predict_proba(X_test)[:,1],
                                         drop_intermediate=False)

plt.plot(fpr_knn, tpr_knn, 'y--',label = 'Knn (area = %0.3f)' % knn_roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc = "lower right")
plt.show()
        



