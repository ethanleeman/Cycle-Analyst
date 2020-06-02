import pickle
import os
import pandas as pd
import networkx as nx
import osmnx as ox
import overpy
import shapely.geometry as geometry
from shapely.ops import linemerge, unary_union, polygonize
import osgeo
from osgeo import ogr
import geopandas as gpd
import fiona
from time import time
import mapclassify
import geoplot
pip install mapclassify
pip install geoplot

name = 'Philadelphia, Pennsylvania, USA'
north = 39.9811
south = 39.9228
east = -75.1417
west = -75.2309
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


ox.settings.useful_tags_path = ['route','lcn','segregated','rcn','rcn_ref','lcn_ref','bridge','tunnel','oneway','lanes','highway','maxspeed','service','access','area','landuse','width','est_width','junction','cycleway:right','cycleway:left','surface','cycleway','cycleway:both']


def load_graph(name,north=None,south=None,east=None,west=None):
    pickle_path = './' +name+'.gpickle'
    if os.path.exists(pickle_path):
        graph = nx.read_gpickle(pickle_path)
    else:
        graph = ox.graph_from_place(name, network_type='bike',simplify=True)
        nx.write_gpickel(G, pickle_path)
    if north is not None:
        return ox.truncate.truncate_graph_bbox(graph,north,south,east,west)
    return graph


def load_polygon(name,query):
    pass
polygon = philly

def load_crashes():
    df_crash = pd.read_csv('./RawData/PHILADELPHIA_1999/CRASH_1999_Philadelphia.csv')
    for i in range(18):
        s = str(i)
        if i < 10:
            s = '0' + str(s)
        path = './RawData/PHILADELPHIA_20' + s +'/CRASH_20'+ s + '_Philadelphia.csv'
        df_to_merge = pd.read_csv(path)
        df_crash = pd.concat([df_crash,df_to_merge])
    return df_crash

def load_traffic():
    return pd.read_csv('./RawData/DVRPC_Bicycle_Counts.csv')

def clean_crashes(df,polygon):
    an = df.copy()
    an = an[an['BICYCLE_COUNT'] > 0]
    in_location = an.apply(lambda r: polygon.contains(geometry.Point(r['DEC_LONG'], r['DEC_LAT'])),axis=1)
    an = an[in_location]
    if north is not None:
        an = an[an['DEC_LONG'] < east]
        an = an[an['DEC_LONG'] > west]
        an = an[an['DEC_LAT'] < north]
        an = an[an['DEC_LAT'] > south]
    return an

def clean_traffic(df,polygon):
    an = df.copy()
    if north is not None:
        an = an[an['X'] < east]
        an = an[an['X'] > west]
        an = an[an['Y'] < north]
        an = an[an['Y'] > south]
    in_location = an.apply(lambda r: polygon.contains(geometry.Point(r['X'], r['Y'])),axis=1)
    an = an[in_location]
    return an


def exploder_one_hot(df,column_name_list):
    to_concat = [df.copy()]
    for col in column_name_list:
        to_concat.append(df[col].apply(lambda x: [x]).str.join('|').str.get_dummies().add_prefix(col + ":"))
        #return df[col].apply(lambda x: [x]).str.join('|').str.get_dummies()
    return pd.concat(to_concat,axis=1).drop(column_name_list,axis=1)

def edge_featurizer(df,column_name_list):
    an = exploder_one_hot(df,column_name_list)
    an['x'] = an.apply(lambda r: r.geometry.centroid.x, axis=1)
    an['y'] = an.apply(lambda r: r.geometry.centroid.y, axis=1)
    return an





def traffic_func(r,G):
    return ox.get_nearest_edge(G,(r['Y'],r['X']),return_geom = True,return_dist = True)


def give_each_traffic_an_edge(df_traffic,G):
    df_traffic['closest_edge'] = df_traffic.apply(lambda x : traffic_func(x,G),axis=1)


column_name_list = ['highway', 'surface', 'segregated', 'service', 'cycleway', 'lanes', 'cycleway:right', 'maxspeed', 'cycleway:left', 'access', 'lcn',
        'tunnel', 'bridge', 'rcn_ref', 'width', 'junction']


philly
gedges = gpd.Dataframe({shape: philly})
df_traffic = clean_traffic(load_traffic(),philly)
df_crashes = clean_crashes(load_crashes(),philly)

df_traffic.columns

G = load_graph(name,north,south,east,west)
G_undirected = G.to_undirected()
df_nodes = ox.graph_to_gdfs(G_undirected,edges=False)
df_edges = ox.graph_to_gdfs(G_undirected,nodes=False)
df_edges_with_features = edge_featurizer(df_edges,column_name_list)
df_edges_with_features['oneway'] = df_edges_with_features['oneway']*1.0

t = time()
give_each_traffic_an_edge(df_traffic,G_undirected)
print(time()-t)
df_traffic['closest_edge']
df_traffic['closest_edge_poly'] = df_traffic.apply(lambda x: x['closest_edge'][3], axis=1)
df_traffic['closest_edge_u'] = df_traffic.apply(lambda x: x['closest_edge'][0], axis=1)
df_traffic['closest_edge_v'] = df_traffic.apply(lambda x: x['closest_edge'][1], axis=1)
gdf_traffic = gpd.GeoSeries(df_traffic['closest_edge_poly'])
gdf_traffic.plot()
k
df_traffic.columns
df_traffic.groupby(['closest_edge_u','closest_edge_v','setdate']).get_group((109729474,109729486,'2010/09/02 00:00:00+00'))
df_traffic_grouped = df_traffic.groupby(['closest_edge_u','closest_edge_v','setdate']).agg({'road':'first', 'X':'first','Y':'first', 'setyear':'first', 'aadb':'sum', 'closest_edge':'first', 'closest_edge_poly':'first', 'closest_edge_u':'first', 'closest_edge_v':'first'})
gdf_traffic_grouped = gpd.GeoDataFrame(df_traffic_grouped[['closest_edge_poly','aadb','road']])
gdf_traffic_grouped['geometry'] = gdf_traffic_grouped['closest_edge_poly']
gdf_traffic_grouped.describe()
gdf_traffic_grouped[gdf_traffic_grouped['aadb'] > 3000]

gdf_traffic_grouped[gdf_traffic_grouped['aadb'] > 3000].plot()
gdf_traffic_grouped.plot()

scheme = mapclassify.Quantiles(gdf_traffic_grouped['aadb'],k=5)
geoplot.choropleth(gdf_traffic_grouped, hue=gdf_traffic_grouped['aadb'],scheme=scheme,cmap='Greens')

df_traffic.describe()
df_traffic.count()
ox.get_nearest_edge(G,(r['Y'],r['X']),return_geom = True,return_dist = True)
df_traffic.apply(lambda x : traffic_func(x,G))


df_edges_with_features.columns
df_edges_with_features.head()

df_traffic.head()

edges= ox.graph_to_gdfs(G_undirected,nodes=False)
edges.count()
nx.edges(G_undirected,7244251137)
nx.edges(G,7244251137)
help(G_undirected.has_edge)
G_undirected.edges()



fig,ax = ox.plot_graph(G_undirected, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='w', edge_linewidth=1,use_geom=True, axis_off=False,show=False, close=False)
#ax=df_crashes.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax)
ax = df_traffic.plot(title='Plotting the 3 Main Datasets',kind='scatter',x='X',y='Y',s=1,c='r',label='Traffic Study',fig=fig,ax=ax)
