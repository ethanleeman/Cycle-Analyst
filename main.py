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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

pip install mapclassify
pip install geoplot





ox.settings.useful_tags_path = ['route','lcn','segregated','rcn','rcn_ref','lcn_ref','bridge','tunnel','oneway','lanes','highway','maxspeed','service','access','area','landuse','width','est_width','junction','cycleway:right','cycleway:left','surface','cycleway','cycleway:both']




polygon = philly

edges = pd.read_pickle('./edges.pkl')

edges.head()








def beta_values(mean,std):
    alpha = ((1-mean)/std/std -(1 / mean))* mean * mean
    beta = alpha*(1/mean - 1)
    return alpha,beta


column_name_list = ['highway', 'surface', 'segregated', 'service', 'cycleway', 'lanes', 'cycleway:right', 'maxspeed', 'cycleway:left', 'access', 'lcn',
        'tunnel', 'bridge', 'rcn_ref', 'width', 'junction']




#reg = LinearRegression().fit(traffic_x.sort_index(axis=1),traffic_y)
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(traffic_x,traffic_y)
np.array(traffic_y)/regr.predict(traffic_x.sort_index(axis=1))

df_edges_with_features.columns
df_edges_with_features['setyear'] = 2018
df_edges_with_features.columns
traffic_as_input = df_edges_with_features.drop(['u','v','key','osmid','geometry'],axis=1).sort_index(axis=1)
df_edges['aadb_predictions'] = regr.predict(traffic_as_input)

G_with_traffic_weights = ox.graph_from_gdfs(df_nodes,df_edges)
#ox.plot.get_edge_colors_by_attr(G_with_traffic_weights,'aadb_predictions')

ec = ox.plot.get_edge_colors_by_attr(G_with_traffic_weights, 'aadb_predictions', cmap='plasma',num_bins=20)
fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.03,edge_linewidth=df_edges['aadb_predictions']*.002,node_alpha = 0.1,node_color='k', bgcolor='w',use_geom=True, axis_off=False,show=False, close=False)
fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.03,edge_linewidth=1,edge_color=ec,node_alpha = 0.5,node_color='k', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)

#
#
# column_name_list = ['highway', 'surface', 'segregated', 'service', 'cycleway', 'lanes', 'cycleway:right', 'maxspeed', 'cycleway:left', 'access', 'lcn',
#         'tunnel', 'bridge', 'rcn_ref', 'width', 'junction']
#
#
# philly
#
# ## loading data
# df_traffic = clean_traffic(load_traffic(),philly)
# df_crashes = clean_crashes(load_crashes(),philly)
# G = load_graph(name,north,south,east,west)
# G_undirected = G.to_undirected()
# df_nodes = ox.graph_to_gdfs(G_undirected,edges=False)
# df_edges = ox.graph_to_gdfs(G_undirected,nodes=False)
# df_edges_with_features = edge_featurizer(df_edges,column_name_list)
# df_edges_with_features['oneway'] = df_edges_with_features['oneway']*1.0
#
#
# # give each traffic an edge, take a while (10min)
# t = time()
# give_each_traffic_an_edge(df_traffic,G_undirected)
# print(time()-t)
#
#
# df_traffic['closest_edge']
# df_traffic['closest_edge_poly'] = df_traffic.apply(lambda x: x['closest_edge'][3], axis=1)
# df_traffic['u'] = df_traffic.apply(lambda x: x['closest_edge'][0], axis=1)
# df_traffic['v'] = df_traffic.apply(lambda x: x['closest_edge'][1], axis=1)
# gdf_traffic = gpd.GeoSeries(df_traffic['closest_edge_poly'])
#
# df_traffic.head(1)
# df_traffic.columns
#
# df_traffic_grouped = df_traffic.groupby(['u','v','setdate']).agg({'road':'first','X':'first','Y':'first', 'setyear':'first', 'aadb':'sum', 'closest_edge':'first', 'closest_edge_poly':'first'})
# df_traffic_grouped['key'] = 0
# df_traffic_grouped = df_traffic_grouped.reset_index()
#
# df_traffic_grouped_with_features = pd.merge(df_traffic_grouped,df_edges_with_features, how = 'left', on=['u','v','key'])
# df_traffic_grouped_with_features.columns
# df_traffic_grouped_with_features.head()
# df_traffic_grouped_with_features = df_traffic_grouped_with_features.drop(['road','X','Y','closest_edge','u','v','setdate','closest_edge_poly'],axis=1)
# df_traffic_grouped_with_features.columns
#
# df_traffic_grouped_with_features_altered = df_traffic_grouped_with_features[df_traffic_grouped_with_features['setyear'] < 2019]
# df_traffic_grouped_with_features_altered['aadb'].hist()
#
# df_traffic_grouped_with_features_altered.groupby('setyear')['aadb'].mean()[2010]
#
# def scale_traffic_for_year_2017(df_traffic):
#     s = df_traffic.groupby('setyear')['aadb'].mean()
#     amt = s[2017]
#     df_traffic['aadb_year_adjusted'] = df_traffic.apply(lambda x: x['aadb']*amt/s[x['setyear']],axis=1)
#
# scale_traffic_for_year_2017(df_traffic_grouped_with_features_altered)
#
#
#
# traffic_x = df_traffic_grouped_with_features_altered.drop(['key','aadb','osmid','geometry'],axis=1)
# traffic_y = df_traffic_grouped_with_features_altered[['aadb','setyear']]
#
#
# traffic_x_train = traffic_x[traffic_x['setyear'] != 2017]
# traffic_y_train = traffic_y[traffic_y['setyear'] != 2017]
#
# traffic_x_train.columns
# #traffic_y_train = traffic_y_train.drop('setyear',axis=1)
# #traffic_x_train = traffic_x_train.drop('aadb',axis=1)
# traffic_x_train = traffic_x_train.sort_index(axis=1)
#
#
# np.ravel(traffic_y_train.head())
#
# ## models for traffic
# #reg = LinearRegression().fit(traffic_x_train.sort_index(axis=1),traffic_y_train)
# regr = RandomForestRegressor(max_depth=5, random_state=0)
# regr.fit(traffic_x_train,np.ravel(traffic_y_train['aadb']))
# np.array(traffic_y_train['aadb'])/regr.predict(traffic_x_train.sort_index(axis=1))
#
# #df_edges_with_features.columns
# df_edges_with_features['setyear'] = 2017
#
# traffic_as_input = df_edges_with_features.drop(['u','v','key','osmid','geometry'],axis=1).sort_index(axis=1)
# df_edges['aadb_predictions'] = regr.predict(traffic_as_input)
#
# G_with_traffic_weights = ox.graph_from_gdfs(df_nodes,df_edges)
# ox.plot.get_edge_colors_by_attr(G_with_traffic_weights,'aadb_predictions').shape()
#
# ec = ox.plot.get_edge_colors_by_attr(G_with_traffic_weights, 'aadb_predictions', cmap='plasma',num_bins=20)
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.03,edge_linewidth=df_edges['aadb_predictions']*.002,node_alpha = 0.1,node_color='k', bgcolor='w',use_geom=True, axis_off=False,show=False, close=False)
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.03,edge_linewidth=1,edge_color=ec,node_alpha = 0.5,node_color='k', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)



#G_with_traffic_weights = ox.utils_graph.graph_from_gdfs(df_nodes,df_traffic_grouped)
#fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=df_nodes['number_of_accidents'],edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)

# G_with_traffic_weights = ox.graph_from_gdfs(df_nodes,df_edges)
#
#
# df_traffic_grouped[df_traffic_grouped['v'] ==109729474 ]
# G_with_traffic_weights.degree(109729474)
# G_with_traffic_weights.degree(109729474,weight='aadb_predictions')
# df_nodes['aadb'] = df_nodes.apply(lambda x: G_with_traffic_weights.degree(x['osmid'],weight='aadb_predictions'), axis=1)
# df_nodes[df_nodes['osmid'] == 109729474]
#
# ## 1999 to 2016 is 17 years, 365 days
# df_nodes['accidents/aadb'] = df_nodes['number_of_accidents'].div(df_nodes['aadb']) /365/17
#
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=df_nodes['number_of_accidents'],edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=df_nodes['accidents/aadb']*10000000,edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)
#
#
# df_nodes['aadb'].hist()
# df_nodes['number_of_accidents'].hist()
# df_nodes['accidents/aadb'].std()
#
# df_nodes['accidents/aadb'].mean()
#
# alpha,beta = beta_values(df_nodes['accidents/aadb'].mean(),df_nodes['accidents/aadb'].std())
# df_nodes['adjusted accidents/aadb'] = (df_nodes['number_of_accidents']+alpha).div(df_nodes['aadb']+alpha+beta)
#
#
# def probability_func(r,df_nodes):
#     u = r['u']
#     v = r['v']
#     return df_nodes.loc[u]['adjusted accidents/aadb'] + df_nodes.loc[v]['adjusted accidents/aadb']
#
# def give_probabilities_to_edges(df_edges,df_nodes):
#     df_edges['probability'] = df_edges.apply(lambda x: probability_func(x,df_nodes),axis=1)
#
#
#
# give_probabilities_to_edges(df_edges,df_nodes)
#
# df_edges.head()
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=df_nodes['adjusted accidents/aadb']*1000000,edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)
#
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges['probability']*70000,use_geom=True, axis_off=False,show=False, close=False)
#
#
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges['length']*.001,use_geom=True, axis_off=False,show=False, close=False)

# df_edges.length.mean()
# df_edges['length'].mean()
# f = df_edges['length'].mean() / df_edges['probability'].mean()
# df_edges['balanced_weight'] = df_edges.apply(lambda x: x['length'] + x['probability']*f*2,axis=1)
# df_edges[['length','probability','balanced_weight']].head(20)
# fig,ax = ox.plot_graph(G_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges['balanced_weight']*.001,use_geom=True, axis_off=False,show=False, close=False)
# df_edges['length'].div(df_edges['balanced_weight']).mean()
final_G = ox.graph_from_gdfs(df_nodes,df_edges)
final_G = final_G.to_undirected()


df_edges_final_one_way.head()
df_edges_final_one_way_with_weights.iloc[-1]['probability_x'] == np.NaN

df_edges.head()

len(df_edges_final_one_way)
len(df_edges)

df_edges_final_one_way = ox.graph_to_gdfs(load_graph(name,north,south,east,west),nodes=False)
df_edges_final_one_way_with_weights = pd.merge(df_edges_final_one_way,df_edges[['u','v','probability','balanced_weight']],how='left',on=['u','v'])
df_edges_swapped = df_edges.copy()
df_edges_swapped['t'] = df_edges_swapped['u']
df_edges_swapped['u'] = df_edges_swapped['v']
df_edges_swapped['v'] = df_edges_swapped['t']
df_edges_final_one_way_with_weights = pd.merge(df_edges_final_one_way_with_weights,df_edges_swapped[['u','v','probability','balanced_weight']],how='left',on=['u','v'])
df_edges_final_one_way_with_weights['probability'] = df_edges_final_one_way_with_weights.fillna(0)[['probability_x','probability_y']].max(axis=1)
df_edges_final_one_way_with_weights['balanced_weight'] = df_edges_final_one_way_with_weights.fillna(0)[['balanced_weight_x','balanced_weight_y']].max(axis=1)

final_G = ox.graph_from_gdfs(df_nodes,df_edges_final_one_way_with_weights)


orig = ox.get_nearest_node(final_G,(south+(north-south)*.5,west+(east-west)*.5))
dest = ox.get_nearest_node(final_G,(south+(north-south)*.2,west+(east-west)*.2))
route_1 = nx.shortest_path(final_G,orig,dest,weight='length')
route_2 = nx.shortest_path(final_G,orig,dest,weight='probability')
route_3 = nx.shortest_path(final_G,orig,dest,weight='balanced_weight')

ox.graph_to_gdfs(final_G,nodes=False)

route_1_length      = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_1,'length'))
route_1_probability = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_1,'probability'))
route_2_length      = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_2,'length'))
route_2_probability = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_2,'probability'))
route_3_length      = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_3,'length'))
route_3_probability = sum(ox.utils_graph.get_route_edge_attributes(final_G,route_3,'probability'))
print('route 1 length: ' + str(route_1_length))
print('route 1 prob  : ' + str(route_1_probability))
print('route 2 length: ' + str(route_2_length))
print('route 2 prob  : ' + str(route_2_probability))
print('route 3 length: ' + str(route_3_length))
print('route 3 prob  : ' + str(route_3_probability))

ox.plot_graph_route(final_G, route_1, route_linewidth =  6, node_size = df_nodes['number_of_accidents']*7, node_color='r', bgcolor ='k',route_color='b')
ox.plot_graph_route(final_G, route_2, route_linewidth =  6, node_size = df_nodes['number_of_accidents']*7, node_color='r', bgcolor ='k',route_color='g')
ox.plot_graph_route(final_G, route_3, route_linewidth =  6, node_size = df_nodes['number_of_accidents']*7, node_color='r', bgcolor ='k',route_color='y')

help(ox.io.save_graph_xml)
ox.io.save_graph_xml(final_G,'./philadelphia.graphxml')
df_nodes.to_pickle('./nodes.pkl',protocol = 4)
df_edges_final_one_way_with_weights.to_pickle('./edges.pkl',protocol = 4)

df_edges['probability_x10000'] = df_edges['probability'] *10000
df_edges.dtypes

nx.shortest_path(final_G, orig, dest, weight = 'probability')

pd.read_pickle('nodes.pkl')




df_traffic_grouped_with_features['setyear'].hist()
df_traffic_test = df_traffic_grouped_with_features[df_traffic_grouped_with_features['setyear'] == 2016]
len(df_traffic_test)
df_crashes_test = df_crashes[df_crashes['CRASH_YEAR'] == 2016]

df_nodes_copy = df_nodes.copy()
number_of_crashes_at_a_node(df_crashes_test,G_with_traffic_weights,df_nodes_copy,prefix = 'test_')
df_nodes_copy['number_of_accidentstest_'].sum()

df_traffic_test['accidents_on_edge'] = df_traffic_test.apply(lambda r: df_nodes_copy.loc[r['u']]['number_of_accidentstest_'] + df_nodes_copy.loc[r['v']]['number_of_accidentstest_'],axis=1)

df_traffic_test

df_nodes_copy.head()
df_traffic_test.head()

#G_for_test = ox.graph_from_gdfs(df_nodes_copy,df_edges)


len(df_crashes_test)
df_crashes_test.head()

df_edges.iloc[0][['u','v']]
ox.graph_to_gdfs(G_with_traffic_weights,edges=False)
u
v

df_edges['accident_prob'] = df_edges.apply


df_nodes['adjusted accidents/aadb'].describe()
len(df_nodes[df_nodes['adjusted accidents/aadb'] < 0.000301]) - len(df_nodes)


df_nodes[(df_nodes['number_of_accidents'] > 0)]['adjusted accidents/aadb'].min()
df_nodes[(df_nodes['adjusted accidents/aadb'] < 0.000191) & (df_nodes['number_of_accidents'] > 0)]

df_nodes.loc[110330155]
df_nodes.loc[110329806]
df_nodes[(df_nodes['adjusted accidents/aadb'] > 0.000181) & (df_nodes['number_of_accidents'] > 0)]

len(df_nodes[df_nodes['number_of_accidents'] == 0 ]) - len(df_nodes)

ax = df_nodes['adjusted accidents/aadb'].hist()
ax.set_xscale('log')
ax.set_yscale('log')

traffic_x
reg.coef_



gdf_traffic.plot()
k
df_traffic.columns
df_traffic.groupby(['closest_edge_u','closest_edge_v','setdate']).get_group((109729474,109729486,'2010/09/02 00:00:00+00'))
gdf_traffic_grouped = gpd.GeoDataFrame(df_traffic_grouped[['closest_edge_poly','aadb','road','setyear']])
gdf_traffic_grouped['geometry'] = gdf_traffic_grouped['closest_edge_poly']
gdf_traffic_grouped
gdf_traffic_grouped[gdf_traffic_grouped['aadb'] > 3000]
fig,ax = ox.plot_graph(G_traffic, node_zorder=2,node_size=0.03,edge_linewidth=df_traffic_grouped['aadb']*.002,node_alpha = 0.1,node_color='k', bgcolor='w',use_geom=True, axis_off=False,show=False, close=False)

gdf_traffic_grouped[gdf_traffic_grouped['aadb'] == 234].plot(ax=ax)

gdf_traffic_grouped[gdf_traffic_grouped['aadb'] == 354]

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

df = pd.DataFrame({'A': [1,2,3],'B':[2,3,4]})
df.A

fig,ax = ox.plot_graph(G_undirected, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='w', edge_linewidth=1,use_geom=True, axis_off=False,show=False, close=False)
#ax=df_crashes.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax)
ax = df_traffic.plot(title='Plotting the 3 Main Datasets',kind='scatter',x='X',y='Y',s=1,c='r',label='Traffic Study',fig=fig,ax=ax)
