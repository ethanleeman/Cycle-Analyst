import osmnx as ox
from DataCleaning import clean_crash_data
import importlib
importlib.reload(DataCleaning)

north = 39.92003
south = 39.91651
east = -75.17989
west = -75.18690

G = ox.graph_from_bbox(north,south,east,west,simplify=True,network_type='bike')


df_crash = pd.read_csv('./RawData/PHILADELPHIA_1999/CRASH_1999_Philadelphia.csv')
for i in range(18):
    s = str(i)
    if i < 10:
        s = '0' + str(s)
    path = './RawData/PHILADELPHIA_20' + s +'/CRASH_20'+ s + '_Philadelphia.csv'
    df_to_merge = pd.read_csv(path)
    df_crash = pd.concat([df_crash,df_to_merge])
df_crash.columns
df_crash = clean_crash_data(df_crash)
df_crash_small = df_crash[df_crash['DEC_LAT'] > south]
df_crash_small = df_crash_small[df_crash_small['DEC_LAT'] < north]
df_crash_small = df_crash_small[df_crash_small['DEC_LONG'] < east]
df_crash_small = df_crash_small[df_crash_small['DEC_LONG'] > west]
len(df_crash_small)

fig,ax = ox.plot_graph(G, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
ax=df_crash_small.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax)

ox.get_nearest_node(G,(39.9180,-75.183),return_dist=True)
gdf_nodes = ox.graph_to_gdfs(G)[0]
gdf_nodes.head()

def crash_func(r,G):
    return ox.get_nearest_node(G,(r['DEC_LAT'],r['DEC_LONG']))

2311 in df_crash_small.head()
def _crash_tag(df_nodes,s_counts):
    if df['osmid']

def tag_crashes(df_crash,G,df_nodes,prefix=''):
    df_crash['nearest_node'] = df_crash.apply(lambda x: crash_func(x,G),axis=1)
    s_counts = df_crash['nearest_node'].value_counts()
    df_nodes['number_of_accidents' + prefix] = df_nodes.apply(lambda x: s_counts[x['osmid']] if x['osmid'] in s_counts.index else 0,axis=1)

tag_crashes(df_crash_small,G,gdf_nodes)
gdf_nodes
df_crash_small
type(df_crash_small.apply(lambda x: crash_func(x,G),axis=1).value_counts())

df_crash_small['DEC_LAT']
