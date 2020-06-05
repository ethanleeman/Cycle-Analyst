import pandas as pd
from DataCleaning import clean_traffic

def traffic_func(r,G):
    return ox.get_nearest_node(G,(r['Y'],r['X']))

def tag_traffic(df_traffic,G,df_nodes,prefix=''):
    df_traffic['nearest_node'] = df_traffic.apply(lambda x: traffic_func(x,G),axis=1)
    s_counts = df_traffic['nearest_node'].value_counts()
    df_nodes['number_of_traffic_studies' + prefix] = df_nodes.apply(lambda x: s_counts[x['osmid']] if x['osmid'] in s_counts.index else 0,axis=1)
