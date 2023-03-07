#!/usr/bin/env python
# coding: utf-8

# # Lending Club Case Study

# ### According to the problem statement in this case study we need to assess and find out the variables which can give us some ide of whether the customer will pay off the loan or if he will default.

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#import the csv file dataset into python
df = pd.read_csv('/Users/nnelson/Documents/LendingClubCaseStudy/loan.csv', low_memory=False)
df.head(10)


# In[3]:


df.shape


# #### Shape function gave us the details that there are 39717 rows and 111 Columns.
# #### It is a big Dataframe with 111 variables and working with so many columns or variables would be difficult
# #### hence we will reduce the number of columns to what we would need for this Case Study
# 

# In[4]:


df.info()


# In[8]:


df.isnull().sum()


# #### Looks like there are several columns which contain only NULL values and hence we would need to remove those columns

# In[10]:


df = df.dropna(axis=1, how='all')
df.isnull().sum()


# In[11]:


df.shape


# #### We still have 57 Columns and that is still a large number of Variables. 
# #### Lets see if they are all important Variables.
# 
# #### As per the Problem Statement we need to find out the probability of loan default when the customer applies for Loan
# #### whether the customer will default or not would need to predicted with the info that we will get when he applies.
# #### Hence we can ignore the columns or variables that we would not have when the customer applies for loan
# #### we will keep only the info that we would have access to when the customer applies for loan

# In[37]:


df1 = df[['loan_amnt','term','int_rate','grade','sub_grade','home_ownership','annual_inc','verification_status','loan_status', 'purpose', 'dti','emp_length','issue_d']]


# In[38]:


df1.head()


# In[39]:


df1.shape


# In[40]:


df1.isnull().sum()


# #### We have about more than 1000 NULL entries in the Emp_Length. This might impact our EDA. Hence it would be better to remove those rows.

# In[41]:


df1 = df1[-df1['emp_length'].isnull()]


# In[42]:


df1.isnull().sum()


# #### Now our data looks clean and simple and it would be easy to do EDA on this sample of data.

# In[43]:


df1.shape


# #### Total Number of Records : 38642

# ### Univariate Analysis

# #### Lets start by doing a Univariate Analysis on Loan Amount first

# In[44]:


df1.loan_amnt.value_counts()


# In[45]:


#The loan amount is varied and its difficult to get an insight from the Variable this way
# lets plot a histogram and see if we can see something

df1.loan_amnt.plot.hist(bins = 20, edgecolor = 'white')
plt.show()


# ##### from the above Histogram we get an idea that most of the loan applications are for amounts 5000 and then it gradually decreases as the loan amount increases. Then we see a small spike at the end at 35000. Least applications are between 30,000 and 35000. Also mostly people apply for a loan in figures like 5000, 10000, 15000, 20000 and so on rather than 16000, 21000 ,31000. So We can see a trend here that people tend to go for a round figure when applying for loan more. 

# In[46]:


# Lets also have a look at the term and see if people take more 36 months loan or if they prefer 60 months.

df1.term.value_counts()


# In[52]:


labels = ['3 Years','5 Years']
plt.pie(df1.term.value_counts(), labels = labels)
plt.show()


# #### The above Pie Chart shows us that almost 1/3rd of the loan applications were for 3 Years.

# In[53]:


# Loan Apllication distributions based on Home Ownership

df1.home_ownership.value_counts()


# In[59]:


labels = ['RENT','MORTGAGE','OWN','OTHER','']
plt.pie(df1.home_ownership.value_counts(), labels = labels)
plt.show()


# #### From the above Pie Chart we can clearly see that a huge chunk of the applications were from the people who either rent or are paying Mortages. There are people who own a house and also have applied but they are few.
# 
# 

# In[60]:


# Let us look at the distribution of Income of the people who have applied for the loans


# In[69]:


df1.annual_inc.describe()


# In[89]:


df1.annual_inc.isnull().sum()


# In[91]:


df1.annual_inc.median()


# In[88]:


df1.annual_inc.max()


# In[92]:


df1.annual_inc.mode()


# In[96]:


plt.boxplot(df1.annual_inc)


# In[97]:


# There are a few applications in which the income decalred is in Millions . The maximum being 6 million

df1[df1.annual_inc > 1000000]


# In[100]:


# Looks like except one loan all the loans taken by the people who declared income as more than 1000000 are fully paid.
# Just for looking at the data more clearly we will consider the ones withc more than 10000000 income as outliers for now

df2 = df1[df1.annual_inc < 1000000]

plt.boxplot(df2.annual_inc)


# In[102]:


df3 = df1[df1.annual_inc < 400000]

plt.boxplot(df3.annual_inc)


# In[104]:


df3.annual_inc.describe()


# In[106]:


df3.annual_inc.median()


# In[107]:


df3.annual_inc.mode()


# #### We can get a idea from all these data that about 50% of the loan applications were from people earning between 40000 to 82000 annually. 

# In[108]:


df4 = df1[df1.annual_inc < 200000]

plt.boxplot(df4.annual_inc)
plt.show()


# In[110]:


# Let us also check about the Verification Status Distribution as well

df1.verification_status.value_counts()


# In[112]:


labels = ['Not Verified','Verified','Source Verified']
plt.pie(df1.verification_status.value_counts(), labels= labels)
plt.show()


# #### A distribution of the verification status shows that there is a big chunk of applications that are not verified and there is also a big chunk which was verified by 3rd party.

# In[113]:


# Let us also check about the Distribution of the Loan Purpose as well

df1.purpose.value_counts()


# In[115]:


pp = pd.DataFrame(df1.purpose.value_counts())


# In[116]:


pp.head()


# In[117]:


pp = pp.reset_index()
pp.columns = ['purpose','no_of_loans']
pp.head()


# In[122]:


pp['percent'] = round(pp['no_of_loans']/38642*100)


# In[123]:


pp.head()


# In[125]:


pp.plot.bar('purpose','percent')


# #### From the above diagram we can see that almost half of the loans were for 'debt_consolidation'  with 'credit card' at a distant second position.

# ##  
# ##  
# ## Segmented Univariate Analysis

# #### Using Segmented Univariate Analysis we will study the relationship between those who Fully Paid and those who defaulted.

# #### In order to see the clear picture we must remove the records where the Loan status is current because we dont know yet if those customers will default or not.

# In[234]:


# removing the Current Loan records

df = df1[df1.loan_status != 'Current']


# In[146]:


df.loan_status.value_counts()


# In[147]:


df.shape


# #### Total records now : 37544

# In[148]:


loan_stat = pd.DataFrame(df.loan_status.value_counts())
loan_stat = loan_stat.reset_index()
loan_stat.columns = ['status','no_of_loans']
loan_stat['percent'] = round(loan_stat['no_of_loans']/37544*100)
loan_stat.plot.bar('status','percent')
plt.show()


# In[149]:


loan_stat


# #### from the above graph we see that about 14% of the total loans which were closed were closed as Charged Off. Which means customers did not pay those loans. Now we have to analyse and see what is the relationship of the status with different variables and if there is any trend that we see

# In[235]:


df.head(10)


# #### Verification Status might play an important role in the data. Let us see how Verification Status and Loan Status co relate

# In[162]:


df_v = df[df['verification_status']=='Verified']
df_v.loan_status.value_counts().plot(kind = 'bar')


# In[163]:


df_v.loan_status.value_counts()


# #### Fully Paid /(Fully Paid + Charged Off)*100 = 83% loans were paid off when they are verified

# In[164]:


df_nv = df[df['verification_status']=='Not Verified']
df_nv.loan_status.value_counts().plot(kind = 'bar')


# In[165]:


df_nv.loan_status.value_counts()


# #### Fully Paid /(Fully Paid + Charged Off)*100 = 87% loans were paid off when they are not verified

# In[166]:


df_sv = df[df['verification_status']=='Source Verified']
df_sv.loan_status.value_counts().plot(kind = 'bar')


# In[167]:


df_sv.loan_status.value_counts()


# #### Fully Paid /(Fully Paid + Charged Off)*100 = 85% loans were paid off when they are Source Verified

# ### Interestingly 87% of the loans that were Not Verified were Fully Paid compared to Verified Loans in which only 83% were paid off and the ones verified by Source 85% were paid off

# In[190]:


df_def = df[df['loan_status'] == 'Charged Off']
df_def['purpose'].value_counts()


# In[191]:


df_purpose_seg = pd.DataFrame(df_def['purpose'].value_counts())
df_purpose_seg = df_purpose_seg.reset_index()
df_purpose_seg.head()


# In[192]:


df_purpose_seg.columns = ['purpose', 'charged_off_loans']


# In[193]:


df_purpose_seg.head()


# In[236]:


df.head()


# In[195]:


df_test = pd.DataFrame(df['purpose'].value_counts())
df_test.head()


# In[196]:


df_test = df_test.reset_index()
df_test.columns = ['purpose','total_loans']
df_test.head()


# In[202]:


df_purpose_seg = df_purpose_seg.merge(df_test, on = 'purpose')


# In[203]:


df_purpose_seg.head(10)


# In[204]:


df_purpose_seg['percent'] = round(df_purpose_seg['charged_off_loans']/df_purpose_seg['total_loans']*100)
df_purpose_seg.head(10)


# In[206]:


df_purpose_seg.plot.bar('purpose','percent')


# #### A whopping 27% of all the loans taken for Small Business were Charged Off making it the most risky type of loan. Second most risky is 'other' category at 16% followed by 'debt_consolidationn', 'medical' and 'moving' at 15%. The most safest loans are 'Credit Card', 'Major Purchase' and 'Wedding'

# In[207]:


df.head(10)


# In[208]:


df_default.home_ownership.value_counts()


# In[209]:


df_homeseg = pd.DataFrame(df_default.home_ownership.value_counts())
df_homeseg = df_homeseg.reset_index()
df_homeseg.columns = ['own_type','charged_off_loans']
df_homeseg.head()


# In[211]:


df_test = pd.DataFrame(df.home_ownership.value_counts())
df_test = df_test.reset_index()
df_test.columns = ['own_type','total_loans']
df_test.head()


# In[212]:


df_homeseg = df_homeseg.merge(df_test, on = 'own_type')
df_homeseg.head()


# In[214]:


df_homeseg['percent'] = round(df_homeseg['charged_off_loans']/df_homeseg['total_loans']*100)
df_homeseg


# In[215]:


df_homeseg.plot.bar('own_type','percent')


# #### The relationship of Home Ownership with the repayment is good as in there is not much difference if the customer owns or rents or if he is paying mortages. They all fall in 14% or 15% category. 'Other' category however stands out at 18% as the most risky one

# # 

# #### In order study and analyse the salary and its relationship we would have to divide it into various buckets

# In[217]:


df.annual_inc.value_counts()


# In[218]:


df.annual_inc.describe()


# In[237]:


# Lets us make the following bins for Annual Income : 20000, 40000, 60000, 80000, 100000, above 100000

df.loc[df['annual_inc'].between(0, 20000, 'right'), 'income_cat'] = '20000'
df.loc[df['annual_inc'].between(20000, 40000, 'right'), 'income_cat'] = '40000'
df.loc[df['annual_inc'].between(40000, 60000, 'right'), 'income_cat'] = '60000'
df.loc[df['annual_inc'].between(60000, 80000, 'right'), 'income_cat'] = '80000'
df.loc[df['annual_inc'].between(80000, 100000, 'right'), 'income_cat'] = '100000'
df.loc[df['annual_inc'].between(100000, 1000000, 'right'), 'income_cat'] = '1000000'
df.loc[df['annual_inc'].between(1000000, 6000000, 'right'), 'income_cat'] = '6000000'


# In[238]:


df.head(20)


# In[239]:


df.income_cat.value_counts()


# In[240]:


#### Cleaning of the emp length for numerical functions

df[['exper','x','y']] = df['emp_length'].str.split(' ', expand = True)


# In[241]:


df


# In[242]:


df.exper.value_counts()


# In[243]:


df['exper'] = df['exper'].replace('10+', 10)


# In[244]:


df['exper'] = df['exper'].replace('<', 1)


# In[245]:


df.head()


# In[246]:


df.exper.value_counts()


# In[251]:


df = df.drop(['x','y'], axis = 1)


# In[252]:


df.head()


# In[253]:


df.exper.value_counts()


# In[270]:


# Let us also fix the date by dividinng the Date into Month and Year
df.issue_d.dtype


# In[271]:


df[['month','year']] = df['issue_d'].str.split('-', expand = True)


# In[272]:


df.head()


# In[273]:


df.describe()


# In[274]:


df['exper']=df['exper'].astype(int)


# In[275]:


df.describe()


# In[276]:


df['year']=df['year'].astype(int)


# In[277]:


df.describe()


# In[278]:


df.year.median()


# In[279]:


df.year.mode()


# In[280]:


df.year.value_counts()


# In[337]:


df.year.value_counts().plot(kind = 'bar')


# In[281]:


df.month.value_counts()


# In[336]:


df.month.value_counts().plot(kind = 'bar')


# In[282]:


df.head(10)


# In[284]:


plt.scatter(df['annual_inc'],df['loan_amnt'])


# In[286]:


plt.boxplot(df['annual_inc'])


# #### For better clarity on distribution of loans with respec to income we would remove the outliers from income with annual income more than 1 million

# In[287]:


df = df[df['annual_inc'] < 1000000]
df.annual_inc.max()


# In[288]:


plt.boxplot(df['annual_inc'])
plt.show()


# #### Income v/s Loan Amount

# In[289]:


plt.scatter(df['annual_inc'],df['loan_amnt'])


# In[290]:


import seaborn as sns


# In[301]:


sns.jointplot(df['annual_inc'],df['loan_amnt'])
plt.show()


# In[294]:


sns.jointplot(df_default['annual_inc'],df_default['loan_amnt'])
plt.show()


# #### Experience v/s Loan Amount

# In[295]:


sns.jointplot(df['exper'],df['loan_amnt'])
plt.show()


# In[297]:


df_default = df[df['loan_status'] == 'Charged Off']
df_paid = df[df['loan_status'] == 'Fully Paid']


# In[299]:


sns.jointplot(df_default['exper'],df_default['loan_amnt'])
plt.show()


# In[302]:


df.head()


# #### Remove the % sign from the interest Rate

# In[317]:


df['int_rate'] = df['int_rate'].str.removesuffix('%')


# In[322]:


df['int_rate'] = df['int_rate'].astype(float)


# In[323]:


df.head(10)


# In[ ]:





# #### Income vs. Loan Payment

# In[325]:


df_paid.income_cat.value_counts()


# In[326]:


df_incseg = pd.DataFrame(df_paid.income_cat.value_counts())
df_incseg = df_incseg.reset_index()
df_incseg.columns = ['income','paid_loans']
df_incseg.head()


# In[329]:


df_incseg2 = pd.DataFrame(df.income_cat.value_counts())
df_incseg2 = df_incseg2.reset_index()
df_incseg2.columns = ['income','total_loans']
df_incseg2.head()


# In[330]:


df_incseg = df_incseg.merge(df_incseg2, on = 'income')


# In[331]:


df_incseg


# In[332]:


df_incseg = df_incseg.drop(['loans'], axis = 1)


# In[333]:


df_incseg['percent'] = round(df_incseg['paid_loans']/df_incseg['total_loans']*100)
df_incseg.head()


# In[334]:


sns.barplot(df_incseg['income'],df_incseg['percent'])


# In[ ]:





# #### Experience vs. Loan Repayment

# In[339]:


df_expseg = pd.DataFrame(df_paid.exper.value_counts())
df_expseg = df_expseg.reset_index()
df_expseg.columns = ['exp','paid_loans']
df_expseg.head()


# In[340]:


df_expseg2 = pd.DataFrame(df.exper.value_counts())
df_expseg2 = df_expseg2.reset_index()
df_expseg2.columns = ['exp','total_loans']
df_expseg2.head()


# In[341]:


df_expseg = df_expseg.merge(df_expseg2, on = 'exp')


# In[342]:


df_expseg.head()


# In[344]:


df_expseg['percent'] = round(df_expseg['paid_loans']/df_expseg['total_loans']*100)
df_expseg.head()


# In[345]:


sns.barplot(df_expseg['exp'],df_expseg['percent'])
plt.show()


# In[346]:


df_expseg.head(10)


# In[ ]:





# In[355]:


df.head(10)


# In[357]:


plt.figure(figsize = [9,7])
sns.boxplot(df['loan_status'], df['loan_amnt'])
plt.show()


# In[358]:


plt.figure(figsize = [9,7])
sns.boxplot(df['loan_status'], df['int_rate'])
plt.show()


# In[360]:


plt.figure(figsize = [9,10])
sns.boxplot(df['loan_status'], df['annual_inc'])
plt.show()


# In[363]:


plt.figure(figsize = [9,7])
sns.boxplot(df['loan_status'], df['dti'])
plt.show()


# ## Heatmaps for comparing more than 2 variables at a time

# In[365]:


### let us create buckets for the loan amount as well

df['loan_cat'] = pd.qcut(df.loan_amnt, [0,0.2,0.4,0.6,0.8,1], ['Very Small','Small','Medium','Large','Very Large'])
df.head()


# In[369]:


df_paid2 = df[df['loan_status'] == 'Fully Paid']


# In[375]:


pd.pivot_table(data = df, index = 'loan_status', columns = 'loan_cat', values = 'annual_inc', aggfunc = np.median)


# In[376]:


inc_loancat = pd.pivot_table(data = df, index = 'loan_status', columns = 'loan_cat', values = 'annual_inc', aggfunc = np.median)


# In[377]:


sns.heatmap(inc_loancat)


# #### We can see a clear relationship to Salary and Loan Amount when it comes to loan repayment. Those with higher income are more likely to repay the loan than the ones with low income. This patterns is same in all types of loan.

# In[379]:


df_default2 = df[df['loan_status'] == 'Charged Off']


# In[385]:


pd.pivot_table(data = df, index = 'verification_status', columns = 'loan_status', values = 'dti')


# #### When it comes to verification, we see that the ones with 'dti' 14 have also fully paid but with 'Non Verified' and 'Source Verified' even the ones with 'dti' of 13.8 and 13 have defaulted.
# 

# In[394]:


pd.pivot_table(data = df, index = 'loan_status', columns = 'loan_cat', values = 'int_rate', aggfunc = np.median)


# In[405]:


pd.pivot_table(data = df, index = 'emp_length', columns = 'loan_status', values = 'loan_amnt', aggfunc = np.median)


# In[406]:


sns.heatmap(pd.pivot_table(data = df, index = 'emp_length', columns = 'loan_status', values = 'loan_amnt', aggfunc = np.median))


# #### 
