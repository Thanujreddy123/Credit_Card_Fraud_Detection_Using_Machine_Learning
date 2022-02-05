import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pylab import rcParams


rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
st.title("CREDIT CARD FRAUD DETECTION")
st.write("\n\n")
st.sidebar.subheader("ADD FILE")
@st.cache
def load_data(nrows):
    data=pd.read_csv('creditcard.csv',nrows=nrows)
    return data
weeklydata=load_data(8000)
st.write(weeklydata)
uploaded_file=st.sidebar.file_uploader(label="Upload your CSV or Excel file.",type=['csv','xlsx'])
st.write("\n\n")


count_classes = pd.value_counts(weeklydata['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
st.pyplot()
st.write("\n\n")


#this if for amount per transaction by class
fraud = weeklydata[weeklydata['Class']==1]
normal = weeklydata[weeklydata['Class']==0]
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()
st.pyplot()
st.write("\n\n")


#this is for Time of transaction vs amoutn by class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
st.pyplot()
st.write("\n\n")


#this is a pie chart
p=len(normal)
q=len(fraud)
slicing=[p,q]
naming=['normal','fraud']
plt.pie(slicing,labels=naming)
plt.title("Pie chart of Transactions")
st.pyplot()
st.write("\n\n")


st.write("There are ",fraud.shape," Fraud transactions and ",normal.shape," normal transaction we took")
st.write("\n")

#here some data if taken form dataset
data1= weeklydata.sample(frac = 0.8000,random_state=1)
st.write("now total it contains",data1.shape)


corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))


#plot heat map
st.write(sns.heatmap(data1[top_corr_features].corr(),annot=True,cmap="RdYlGn"))
st.pyplot()
st.write("\n\n")


#her take fraud and valid
Fraud = data1[data1['Class']==1]
Valid = data1[data1['Class']==0]


#outlier calculation
outlier_fraction = len(Fraud)/float(len(Valid))
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting
target = "Class"
# Define a random state
state = np.random.RandomState(42)


#her x contains all the input variables or features and y contains all the dependent varibles
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))


#here classifiers is a dictionary that contains all the algorithms
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X),
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Random forest":RandomForestClassifier(n_estimators = 200 , random_state=0),
    "Logistic regression":make_pipeline(StandardScaler(), LogisticRegression())
}
x, y = make_classification(random_state=42)
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_sm, Y_sm= oversample.fit_resample(X, Y)


#her splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(X_sm,Y_sm,test_size=0.4,random_state=42)
n_outliers = len(Fraud)


#here in this loops it will train and predict the class
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        clf.fit(X)
        y_pred=clf.fit_predict(X)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != Y).sum()
        # Run Classification Metrics
        st.write("{}: {}".format(clf_name, n_errors))
        st.write("Accuracy Score :")
        st.write(accuracy_score(Y, y_pred))
        st.write("Classification Report :")
        st.write(classification_report(Y, y_pred))
        st.write(confusion_matrix(y_pred, Y))
    elif clf_name == "Isolation Forest":
        clf.fit(X)
        y_pred=clf.predict(X)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != Y).sum()
        # Run Classification Metrics
        st.write("{}: {}".format(clf_name, n_errors))
        st.write("Accuracy Score :")
        st.write(accuracy_score(Y, y_pred))
        st.write("Classification Report :")
        st.write(classification_report(Y, y_pred))
        st.write(confusion_matrix(y_pred, Y))
    elif clf_name=="Random forest":
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        n_errors=(y_pred!=y_test).sum()
        st.write("Random Forest:",n_errors)
        st.write("Accuracy Score :")
        st.write(accuracy_score(y_test,y_pred))
        st.write(classification_report(y_test ,y_pred))
        st.write(confusion_matrix(y_pred,y_test))
        continue
    else :
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        n_errors = (y_pred != y_test).sum()
        n_errors=(y_pred!=y_test).sum()
        st.write("Logistic regression",n_errors)
        st.write("Accuracy Score :")
        st.write(accuracy_score(y_test,y_pred))
        st.write(confusion_matrix(y_pred,y_test))
        st.write(classification_report(y_test ,y_pred))
        continue
if uploaded_file is not None:
    with open('model_pickle','rb') as f:
        mp=pickle.load(f)
        datas = pd.read_csv(uploaded_file, sep=',')
        data2 = datas.sample(frac=1.0, random_state=1)
        columns = data2.columns.tolist()
        # Filter the columns to remove data we do not want
        columns = [c for c in columns if c not in ["Class"]]
        # Store the variable we are predicting
        target = "Class"
        # Define a random state
        state = np.random.RandomState(42)
        X_r = data2[columns]
        Y_r = data2[target]
        # Print the shapes of X & Y
        ypred = mp.predict(X_r)
        st.sidebar.write(data2)
        for i in ypred:
            if i == 0:
                st.sidebar.write("it is a normal trasaction")
            else:
                st.sidebar.write("it is a fraud trasaction")
    st.sidebar.write(mp)