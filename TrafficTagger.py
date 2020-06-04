import pandas as pd
from DataCleaning import clean_traffic

def traffic_func(r,G):
    return ox.get_nearest_node(G,(r['Y'],r['X']))

def tag_traffic(df_traffic,G,df_nodes,prefix=''):
    df_traffic['nearest_node'] = df_traffic.apply(lambda x: traffic_func(x,G),axis=1)
    s_counts = df_traffic['nearest_node'].value_counts()
    df_nodes['number_of_traffic_studies' + prefix] = df_nodes.apply(lambda x: s_counts[x['osmid']] if x['osmid'] in s_counts.index else 0,axis=1)


north = 39.9959
south = 39.9665
east = -75.1272
west = -75.1833

G = ox.graph_from_bbox(north,south,east,west,simplify=True,network_type='bike')
G = ox.graph_from_place('Philadelphia, Pennsylvania, USA', network_type='bike',simplify=True)

df_traffic = pd.read_csv('./RawData/DVRPC_Bicycle_Counts.csv')
df_traffic = clean_traffic(df_traffic)
len(df_traffic)
df_traffic_small = df_traffic[df_traffic['Y'] > south]
df_traffic_small = df_traffic_small[df_traffic_small['Y'] < north]
df_traffic_small = df_traffic_small[df_traffic_small['X'] < east]
df_traffic_small = df_traffic_small[df_traffic_small['X'] > west]

df_traffic[df_traffic['road']== 'yale ave westbound lane' ]
df_traffic.groupby(['X','Y']).count()
df_traffic.groupby(['X','Y']).get_group((-75.977848,39.791619))
df_traffic.groupby(['X','Y']).get_group((-75.977914,39.791624))

df_traffic_small['road'].unique()
df_traffic_small
df_traffic_small.groupby('setdate').get_group('2011/04/22 00:00:00+00')
dir(df_traffic_small.groupby('setdate'))

len(df_traffic)
gdf_nodes.head()


nodes_small = ox.graph_to_gdfs(G,edges=False)
tag_traffic(df_traffic_small,G,nodes_small)
tag_traffic(df_traffic,G,gdf_nodes)
nodes_small['number_of_traffic_studies'].sum()
nodes_small.groupby('number_of_traffic_studies').count()
len(gdf_nodes) - 63882
gdf_nodes['number_of_traffic_studies'].value_counts()


fig,ax = ox.plot_graph(G, node_zorder=2,node_size=0,node_alpha = 1,node_color='k',bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
ax=df_traffic.plot(kind='scatter',x='X',y='Y',s=10,fig=fig,label='Traffic Studies',ax=ax)
df_traffic_small.columns
df_traffic_small

df_traffic_small.groupby(['X','Y']).count()
