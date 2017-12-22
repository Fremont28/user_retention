#Predict if customer returns for a loan 
#12/20/17 
import pandas as pd 
import numpy as np 
cust_ret=pd.read_csv("LoanStats_2017Q1.csv",encoding='latin-1')
cust_ret.columns 

#are customers returning more than once over the last 6 months?
cust_ret['loan_returns']=np.where(cust_ret['open_acc_6m']>1,0,1)
cust_ret['loan_returns']

#assign numeric variable to loan grades
def grade_assign(y):
    if y=="A":
        return 1
    if y=="B":
        return 2
    if y=="C":
        return 3
    if y=="D":
        return 4
    if y=="E":
        return 5
    if y=="F":
        return 6
    if y=="G":
        return 7 

cust_ret['grade_numeric']=cust_ret['grade'].apply(grade_assign)
#predictor and target variables
sub_loans=cust_ret[["int_rate","grade_numeric","installment","loan_returns"]]
sub_loans1=sub_loans.dropna()
sub_loans1.shape #88,255 observations

X=sub_loans1[["int_rate","grade_numeric","installment"]]
y=sub_loans1["loan_returns"]

#1. using logistic regression to predict customer return
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# visualize data
x.describe() 
x.grade.value_counts() 
#percent 'A' grade loans
grade_a=(x['grade']=="A")
y.value_counts() #unique values 
67657/len(y) #76.7% return for another loan source: https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas

#test and train data set
test_size=0.25 
seed=5453
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=test_size,
random_state=seed)
model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
model_lr_result=model_lr.score(X_test,y_test)
print(model_lr_result) #76.3% accuracy

#evaluate using shuffle split cross validation
num_samples=8
test_size=0.25
num_instances=len(X)
seed=53525

kfold=cross_validation.ShuffleSplit(n=num_instances,n_iter=num_samples,test_size=test_size,
random_state=seed)
model_lr=LogisticRegression()
model_lr_results=cross_validation.cross_val_score(model,X,y,cv=kfold)
model_lr_results.mean() #average accuracy 76.7%
model_lr_results.std()  # standard deviation 0.00259

#gaussian naive Bayes classification
num_folds=11
num_instances=len(X)
seed=53525
kfold=cross_validation.KFold(n=num_instances,n_folds=num_folds,random_state=seed)
model=GaussianNB()
results=cross_validation.cross_val_score(model,X,y,cv=kfold)
results.mean() #74.53% mean accuracy
results.std()  #0.016 standard deviation

#visualizations 12/21/17******
import plotly.plotly as py
import plotly.graph_objs as go

#a. 
plot1=[
    go.Bar(
        x=cust_ret['grade'],
        y=cust_ret['int_rate']
    )
]

layout=go.Layout(
    title='Interest Rates Increase with Worse Loan Grades'
)

fig=go.Figure(data=plot1,layout=layout)
url=py.plot(fig,file="pandas1")

#b. 
import plotly.plotly as py
import cufflinks as cf
import pandas as pd

plot2=[
    go.Scatter(
        x=cust_ret['loan_amnt'],
        y=cust_ret['installment'],
    )
]

layout=go.Layout(
    title='Monthly Installments Do Not Increase Interest Rates'
)

fig=go.Figure(data=plot2,layout=layout)
url=py.plot(fig,file="pandas2")

#c.
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

#Dash-- *******
cust_returns=cust_ret[["int_rate","loan_amnt","funded_amnt","grade","addr_state"]]
cust_group=cust_ret.groupby("addr_state")
cust_group.size() 
##########
state_group=cust_group.plot(kind='bar')

##12/19/17 data exploration
cust_ret.columns 

# count grade of loans
cust_ret['grade'].value_counts()

import seaborn as sns 
sns.countplot(x='grade',data=cust_ret,palette='hls')
plt.show() 
plt.savefig('figgy')

#averages
#loan returns by sub grade and grade? 
cust_ret.groupby('sub_grade').mean()
grade_loan_returns=cust_ret.groupby('grade').mean() 
#grade a-0.80 (loan returns), b-0.79, c=0.76, d=0.7, e-0.66 (higher interest rates down the line)

# bar
pd.crosstab(cust_ret.grade,cust_ret.loan_returns).plot(kind='bar')
plt.title('Get a Good Loan, Return for Your Next Loan')
plt.xlabel('Loan Grade')
plt.ylabel('Frequency of Users Returning for Loans')
plt.savefig('xxx1')

#stacked bar 
home_owner=cust_ret.loc[(cust_ret.home_ownership=="RENT") | (cust_ret.home_ownership =="OWN") |]
table=pd.crosstab(home_owner.home_ownership,home_owner.grade)
table.div(table.sum(1).astype('float'),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('Ownership')
plt.ylabel('Count')
plt.title('Home Owners Get Slightly Better Loans Than Renters')
plt.savefig('xxy')