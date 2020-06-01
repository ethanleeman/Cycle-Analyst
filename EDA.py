
import pandas as pd
import osmnx as ox
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from shapely.geometry import Point, Polygon
import overpy
import networkx as nx
import overpy
import shapely.geometry as geometry
from shapely.ops import linemerge, unary_union, polygonize
#%matplotlib inline
ox.config(use_cache=True, log_console=True)



query = """[out:json][timeout:25];
rel(188022);
out body;
>;
out skel qt; """
api = overpy.Overpass()
result = api.query(query)

lss = [] #convert ways to linstrings

for ii_w,way in enumerate(result.ways):
    ls_coords = []

    for node in way.nodes:
        ls_coords.append((node.lon,node.lat)) # create a list of node coordinates

    lss.append(geometry.LineString(ls_coords)) # create a LineString from coords


merged = linemerge([*lss]) # merge LineStrings
borders = unary_union(merged) # linestrings to a MultiLineString
polygons = list(polygonize(borders))
philly = geometry.MultiPolygon(polygons)

philly.contains(geometry.Point(-147.7798220, 64.8564400))
philly


df_crash = pd.read_csv('./RawData/PHILADELPHIA_1999/CRASH_1999_Philadelphia.csv')
for i in range(18):
    s = str(i)
    if i < 10:
        s = '0' + str(s)
    path = './RawData/PHILADELPHIA_20' + s +'/CRASH_20'+ s + '_Philadelphia.csv'
    df_to_merge = pd.read_csv(path)
    df_crash = pd.concat([df_crash,df_to_merge])



df_crash = clean_crash_data(df_crash)

len(df_crash)
df_crash.columns

ax=df_crash.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=0.5)

df_crash['DEC_LONG'].min()
df_crash['DEC_LONG'].max()
df_crash['DEC_LAT'].min()
df_crash['DEC_LAT'].max()

df_traffic = pd.read_csv('./RawData/DVRPC_Bicycle_Counts.csv')
df_traffic.head()
df_traffic.columns
df_traffic.describe()
df_traffic.plot(kind='scatter',x='X', y='Y',c='g')

help(df_traffic.apply)

df_traffic_clean = clean_traffic(df_traffic)
len(df_traffic_clean)

df_traffic_clean.plot(kind='scatter',x='X',y='Y',s=0.7,c='g')

ax=df_crash.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=0.1)
df_traffic_clean.plot(kind='scatter',x='X',y='Y',s=0.1,c='r',ax=ax)


df_traffic_clean['aadb'].hist()

nx.algorithms.edge_boundary(G)

# this takes a while
type(G)

G = ox.graph_from_place('Philadelphia, Pennsylvania, USA', network_type='bike')
fig, ax = ox.plot_graph(G, node_zorder=2,node_size=0.03, node_color='k', bgcolor='w', use_geom=True, axis_off=False)

fig,ax = ox.plot_graph(G, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
ax=df_crash.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax)
ax = df_traffic_clean.plot(title='Plotting the 3 Main Datasets',kind='scatter',x='X',y='Y',s=1,c='r',label='Traffic Study',fig=fig,ax=ax)



len(G)
ox.stats.basic_stats(G)



gdf_edge = ox.utils_graph.graph_to_gdfs(G,nodes=False)
import geopandas
dir(gdf_edge)
pd.set_option("display.max_rows", 101)
len(gdf_edge)
gdf_edge.maxspeed.apply(lambda x: [x]).str.join('|').str.get_dummies()
df_edge = pd.DataFrame(gdf_edge)
len(df_edge.explode('service'))
df_edge.explode('service')


df_edge_exploded = exploder(df_edge)
len(df_edge_exploded)
df_edge.columns
df_edge_exploded.maxspeed.unique()

df_crash.columns
