
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data.data_cleaning import *

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('data/churn_train.csv')
dat = data.copy()

dat.head()

dat = create_target(dat)
dat.head()

dat.isnull().sum()/dat.shape[0]

dat.describe()

df = clean_data(dat)

df.info()

df.head()

dat.plot(y = 'churn', kind = 'hist')

dat.plot(y = 'avg_rating_of_driver', kind = 'hist')

signup_cnt = df.groupby('signup_date').size().reset_index(name='cnt').sort_values(by='signup_date')
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(signup_cnt['signup_date'],signup_cnt['cnt'],'o--')
ax.set_title('count over signup time')
ax.set_xlabel('signup date')
fig.autofmt_xdate(rotation=90)
ax.set_ylabel('Number of signups')

signup_cnt = df.groupby('last_trip_date').size().reset_index(name='cnt').sort_values(by='last_trip_date')
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(signup_cnt['last_trip_date'],signup_cnt['cnt'],'o--')
ax.set_title('count over lasttrip time')
ax.set_xlabel('last_trip_date')
fig.autofmt_xdate(rotation=90)
ax.set_ylabel('Number of last_trip')

dff = df.drop(['city','last_trip_date','phone','luxury_car_user','signup_date','avg_rating_by_driver','avg_rating_of_driver'],axis=1)
dff['churn'] = dff['churn'].astype(bool).astype(str)
sns.pairplot(dff, hue="churn", palette="husl", plot_kws=dict(alpha=0.5))

sns.pairplot(dff[['avg_surge','surge_pct','churn']], hue="churn", palette="husl",plot_kws=dict(alpha=0.5), aspect=1.0)

grouped = df[['city', 'churn']].groupby('city').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='city', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of Cities', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='city', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate per City', fontsize=16)
plt.tight_layout()
plt.show()

# Visualization of different cities
grouped = df[['phone', 'churn']].groupby('phone').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='phone', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of phone device', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='phone', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate per phone device', fontsize=16)
plt.tight_layout()
plt.show()

grouped = df[['luxury_car_user', 'churn']].groupby('luxury_car_user').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='luxury_car_user', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of luxury_car_user', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='luxury_car_user', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate luxury_car_user', fontsize=16)
plt.tight_layout()
plt.show()

grouped = df[['trips_in_first_30_days', 'churn']].groupby('trips_in_first_30_days').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='trips_in_first_30_days', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of trips_in_first_30_days', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='trips_in_first_30_days', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate per trips_in_first_30_days', fontsize=16)
plt.tight_layout()
plt.show()

grouped = df[['avg_rating_by_driver', 'churn']].groupby('avg_rating_by_driver').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='avg_rating_by_driver', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of avg_rating_by_driver', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='avg_rating_by_driver', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate per avg_rating_by_driver', fontsize=16)
plt.tight_layout()
plt.show()

df_test = pd.read_csv('data/churn.csv')
np.mean(df_test['avg_rating_by_driver'])

grouped = df[['avg_rating_of_driver', 'churn']].groupby('avg_rating_of_driver').mean().reset_index()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='avg_rating_of_driver', hue='churn', data=df, ax=ax[0])
ax[0].set_title('Count Plot of avg_rating_of_driver', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='avg_rating_of_driver', y='churn', data=df, ax=ax[1]);
ax[1].set_title('Mean Churn Rate per avg_rating_of_driver', fontsize=16)
plt.tight_layout()
plt.show()

df_test = pd.read_csv('data/churn.csv')
np.mean(df_test['avg_rating_of_driver'])

df.plot(y = 'surge_pct', kind = 'hist')

df.plot(y = 'avg_surge', kind = 'hist')

