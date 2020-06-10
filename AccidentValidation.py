import pandas as pd
import itertools
from time import time
import random

def number_of_crashes_at_a_node(df_crash,df_nodes,prefix=''):
    s_counts = df_crash['nearest_node'].value_counts()
    df_nodes[prefix + 'number_of_accidents'] = df_nodes.apply(lambda x: s_counts[x['osmid']] if x['osmid'] in s_counts.index else 0,axis=1)

def beta_values(mean,std):
    alpha = ((1-mean)/std/std -(1 / mean))* mean * mean
    beta = alpha*(1/mean - 1)
    return alpha,beta

regr
df_nodes.head()
df_bike_accidents_in_region_features['CRASH_YEAR']

df_bike_accidents_in_region_features = df_bike_accidents_in_region[['CRASH_YEAR','nearest_node']]
train_test_split(X,y,test_size=1/479, random_state=1)


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


distances = []
distances2 = []
len(years)
years = range(1999,2018)
num_training_years = 10
corr = []
t = time()
count = 0
while count < 10:
#for i in itertools.combinations(years,num_training_years):
    i = random_combination(years,num_training_years)
    count += 1
    train = df_bike_accidents_in_region_features[df_bike_accidents_in_region_features['CRASH_YEAR'].isin(i)]
    test = df_bike_accidents_in_region_features[~df_bike_accidents_in_region_features['CRASH_YEAR'].isin(i)]
    df_nodes_curr = df_nodes.copy()
    number_of_crashes_at_a_node(train,df_nodes_curr)
    df_nodes_curr['accidents/aadb'] = df_nodes_curr['number_of_accidents'].div(df_nodes_curr['aadb'] * num_training_years * 365)
    mean = df_nodes_curr['accidents/aadb'].mean()
    std = df_nodes_curr['accidents/aadb'].std()
    alpha,beta = beta_values(mean,std)
    df_nodes_curr['accidents/aadb adjusted'] = (df_nodes_curr['number_of_accidents']+alpha).div(df_nodes_curr['aadb']* num_training_years * 365 +alpha+beta)
    number_of_crashes_at_a_node(test,df_nodes_curr,prefix='test_')
    df_nodes_curr['test_accidents/aadb'] = df_nodes_curr['test_number_of_accidents'].div(df_nodes_curr['aadb'] * (len(years) - num_training_years)*365)
    distance = df_nodes_curr.cov().loc['test_accidents/aadb']['accidents/aadb adjusted']/ np.sqrt(df_nodes_curr.cov().loc['test_accidents/aadb']['test_accidents/aadb'] * df_nodes_curr.cov().loc['accidents/aadb adjusted']['accidents/aadb adjusted'])
    distance2 = df_nodes_curr.cov().loc['test_accidents/aadb']['accidents/aadb']/ np.sqrt(df_nodes_curr.cov().loc['test_accidents/aadb']['test_accidents/aadb'] * df_nodes_curr.cov().loc['accidents/aadb']['accidents/aadb'])
    distance3 = df_nodes_curr.cov().loc['test_accidents/aadb']['aadb']/ np.sqrt(df_nodes_curr.cov().loc['test_accidents/aadb']['test_accidents/aadb'] * df_nodes_curr.cov().loc['aadb']['aadb'])

    distances.append(distance)
    distances2.append(distance2)
    print(distance, flush=True)
    print(distance2, flush=True)
    print(distance3, flush=True)


print(pd.Series(distances).mean())
print(pd.Series(distances2).mean())
print(time()-t)
pd.Series(distances).mean()

df_nodes_curr.cov().loc['test_accidents/aadb']['accidents/aadb adjusted']/ np.sqrt(df_nodes_curr.cov().loc['test_accidents/aadb']['test_accidents/aadb'] * df_nodes_curr.cov().loc['test_accidents/aadb']['test_accidents/aadb'])
