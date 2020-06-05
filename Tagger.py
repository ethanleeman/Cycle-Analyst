import osmnx as ox
import pandas as pd

def traffic_func(r,G):
    return ox.get_nearest_edge(G,(r['Y'],r['X']),return_geom = True,return_dist = True)

def give_each_traffic_an_edge(df_traffic,G):
    return df_traffic.apply(lambda x : traffic_func(x,G),axis=1)

def crash_func(r,G):
    return ox.get_nearest_node(G,(r['DEC_LAT'],r['DEC_LONG']))

def tag_crashes(df_crash,G):
    return df_crash.apply(lambda x: crash_func(x,G),axis=1)

def number_of_crashes_at_a_node(df_crash,df_nodes,prefix=''):
    s_counts = df_crash['nearest_node'].value_counts()
    df_nodes[prefix + 'number_of_accidents'] = df_nodes.apply(lambda x: s_counts[x['osmid']] if x['osmid'] in s_counts.index else 0,axis=1)
