import DataLoader
import DataCleaner
import Tagger
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import importlib
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
importlib.reload(DataLoader)
importlib.reload(DataCleaner)
importlib.reload(Tagger)


## name of region to save graph and not ask OSM multiple times
## osm query that is that place
name = 'Philadelphia, Pennsylvania, USA'
query = """[out:json][timeout:25];
rel(188022);
out body;
>;
out skel qt; """

## specific region of interest
north = 39.9811
south = 39.9228
east = -75.1417
west = -75.2309

## Tags we want as features in our traffic model
ox.settings.useful_tags_path = ['route','lcn','segregated','rcn','rcn_ref',\
            'lcn_ref','bridge','tunnel','oneway','lanes','highway','maxspeed',\
            'service','access','area','landuse','width','est_width','junction',\
            'cycleway:right','cycleway:left','surface',\
            'cycleway','cycleway:both' ]

## Tags we want to do 1-hot encoding
column_name_list = ['highway', 'surface', 'segregated', 'service', 'cycleway',\
            'lanes', 'cycleway:right', 'maxspeed', 'cycleway:left', 'access', \
            'lcn', 'tunnel', 'bridge', 'rcn_ref', 'width', 'junction']

## Load all relevent data
df_all_traffic_studies = DataLoader.load_traffic()
df_all_traffic_accidents = DataLoader.load_crashes()



G = DataLoader.load_graph(name,north,south,east,west)

polygon = DataLoader.load_polygon(name,query)

# Using undirected graph to tag traffic studies
G_undirected = G.to_undirected()
df_nodes = ox.graph_to_gdfs(G_undirected,edges=False)
df_edges_undirected = ox.graph_to_gdfs(G_undirected,nodes=False)
df_edges_directed = ox.graph_to_gdfs(G,nodes=False)

df_edges_directed.head()

'highway:cycleway', 'highway:living_street', 'highway:path',
       'highway:pedestrian', 'highway:primary', 'highway:primary_link',
       'highway:residential', 'highway:secondary', 'highway:secondary_link',
       'highway:service', 'highway:tertiary', 'highway:tertiary_link',
       'highway:track', 'highway:trunk', 'highway:trunk_link',
       'highway:unclassified'

ec = [
  'blue' if data['highway']=='tertiary_link'
  else 'green' if data['highway']=='residential'
  else 'yellow' if data['highway']=='tertiary'
  else 'orange' if data['highway']=='secondary'
  else 'red' if data['highway']=='primary'
  else 'blue' for u, v, key, data in G.edges(keys=True, data=True)]

#this works, but is static
ox.plot_graph(G,fig_height=8,fig_width=8,node_size=0, edge_color=ec)

# Run cleaning Methods
df_bike_accidents_in_region = DataCleaner.clean_crashes(df_all_traffic_accidents,polygon,north,south,east,west)
df_traffic_studies_in_region = DataCleaner.clean_traffic(df_all_traffic_studies,polygon,north,south,east,west)
df_undirected_edges_with_features = DataCleaner.edge_featurizer(df_edges_undirected,column_name_list)

df_undirected_edges_with_features.columns

len(df_traffic_studies_in_region)
len(df_bike_accidents_in_region)

## some plots
fig,ax = ox.plot_graph(G_undirected, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=0.4,use_geom=True, axis_off=False,show=False, close=False)
ax=df_bike_accidents_in_region.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax,color='r')
ax = df_traffic_studies_in_region.plot(title='Plotting the 3 Main Datasets',kind='scatter',x='X',y='Y',s=3,c='y',label='Traffic Study',fig=fig,ax=ax)

df_traffic_studies_in_region.setyear.describe()
df_traffic_studies_in_region.plot.scatter(x='setyear',y = 'aadb')


df_bike_accidents_in_region.plot.scatter(x='DEC_LONG',y='DEC_LAT',s=0.5)

# This takes some time (10min on my local machine)
df_traffic_studies_in_region['closest_edge'] = Tagger.give_each_traffic_an_edge(df_traffic_studies_in_region,G_undirected)
df_save1 = df_traffic_studies_in_region.copy()
df_save1.to_csv('./CleanedData/save1')
df_traffic_studies_in_region = df_save1.copy()


df_traffic_studies_in_region['closest_edge_poly'] = df_traffic_studies_in_region.apply(lambda x: x['closest_edge'][3], axis=1)
df_traffic_studies_in_region['u'] = df_traffic_studies_in_region.apply(lambda x: x['closest_edge'][0], axis=1)
df_traffic_studies_in_region['v'] = df_traffic_studies_in_region.apply(lambda x: x['closest_edge'][1], axis=1)
gdf_traffic = gpd.GeoSeries(df_traffic_studies_in_region['closest_edge_poly'])

df_traffic_grouped = df_traffic_studies_in_region.groupby(['u','v','setdate']).agg(
                        {'road':'first', 'X':'first','Y':'first', 'setyear':'first',\
                         'aadb':'sum', 'closest_edge':'first', 'closest_edge_poly':'first'})
df_traffic_grouped['key'] = 0
df_traffic_grouped = df_traffic_grouped.reset_index()


df_traffic_grouped_with_features = pd.merge(df_traffic_grouped,df_undirected_edges_with_features, how = 'left', on=['u','v','key'])
df_traffic_grouped_with_features = df_traffic_grouped_with_features.drop(['road','X','Y','closest_edge','setdate','closest_edge_poly'],axis=1)
df_traffic_grouped_with_features = df_traffic_grouped_with_features[df_traffic_grouped_with_features['setyear'] < 2019]


df_traffic_grouped_with_features_no_encoding = pd.merge(df_traffic_grouped,df_edges_undirected, how = 'left', on=['u','v','key'])
df_traffic_grouped_with_features_no_encoding = df_traffic_grouped_with_features_no_encoding.drop(['road','closest_edge','setdate','closest_edge_poly'],axis=1)
df_traffic_grouped_with_features_no_encoding = df_traffic_grouped_with_features_no_encoding[df_traffic_grouped_with_features_no_encoding['setyear'] < 2019]

df_traffic_grouped_with_features_no_encoding.head(1)


df_traffic_grouped_with_features.to_csv('./CleanedData/traffic_studies_with_features')
df_traffic_grouped_with_features_no_encoding.to_csv('./CleanedData/traffic_studies_with_features_no_encoding')

## train/test split
#df_traffic_grouped_with_features_train = df_traffic_grouped_with_features[df_traffic_grouped_with_features['setyear'] != 2016]
df_traffic_grouped_with_features_train = df_traffic_grouped_with_features

df_traffic_grouped_with_features_train.describe()

df_undirected_edges_with_features.columns
df_edges_directed.drop(['u','v','key','osmid'],axis=1).count() / 25256

df_traffic_grouped_with_features_train.head()
# set up for model
traffic_x = df_traffic_grouped_with_features_train.drop(['u','v','key','aadb','osmid','geometry'],axis=1)
traffic_y = df_traffic_grouped_with_features_train['aadb']


# Traffic Model
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(traffic_x,traffic_y)
np.array(traffic_y)/regr.predict(traffic_x.sort_index(axis=1))

## Apply model to edges in network
df_undirected_edges_with_features['setyear'] = 2018
traffic_as_input = df_undirected_edges_with_features.drop(['u','v','key','osmid','geometry'],axis=1).sort_index(axis=1)
df_edges_undirected['aadb_predictions'] = regr.predict(traffic_as_input)

G_undirected_with_traffic_weights = ox.graph_from_gdfs(df_nodes,df_edges_undirected)


## uncomment for traffic model EDA
ec = ox.plot.get_edge_colors_by_attr(G_undirected_with_traffic_weights, 'aadb_predictions', cmap='plasma',num_bins=20)
fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=0.03,edge_linewidth=1,edge_color=ec,node_alpha = 0.5,node_color='k', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)

# Also will take some time
df_bike_accidents_in_region['nearest_node'] = Tagger.tag_crashes(df_bike_accidents_in_region,G)

df_bike_accidents_in_region.to_csv('./CleanedData/accidents_with_nodes')
df_save2 = df_bike_accidents_in_region.copy()
df_save2.to_csv('./CleanedData/save2')
df_bike_accidents_in_region = df_save2.copy()

df_accidents_train = df_bike_accidents_in_region

Tagger.number_of_crashes_at_a_node(df_accidents_train,df_nodes)

G_undirected_with_traffic_weights = ox.utils_graph.graph_from_gdfs(df_nodes,df_edges_undirected)
fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=df_nodes['number_of_accidents'],edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)


df_nodes['aadb'] = df_nodes.apply(lambda x: G_undirected_with_traffic_weights.degree(x['osmid'],weight='aadb_predictions'), axis=1)

## 1999 to 2016 is 17 years, 365 days
df_nodes['accidents/aadb'] = df_nodes['number_of_accidents'].div(df_nodes['aadb']) /365/17

fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=df_nodes['accidents/aadb']*10000000,edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)


def beta_values(mean,std):
    alpha = ((1-mean)/std/std -(1 / mean))* mean * mean
    beta = alpha*(1/mean - 1)
    return alpha,beta



alpha,beta = beta_values(df_nodes['accidents/aadb'].mean(),df_nodes['accidents/aadb'].std())
df_nodes['adjusted accidents/aadb'] = (df_nodes['number_of_accidents']+alpha).div(df_nodes['aadb']+alpha+beta)


def probability_func(r,df_nodes):
    u = r['u']
    v = r['v']
    return df_nodes.loc[u]['adjusted accidents/aadb'] + df_nodes.loc[v]['adjusted accidents/aadb']

def give_probabilities_to_edges(df_edges,df_nodes):
    df_edges['probability'] = df_edges.apply(lambda x: probability_func(x,df_nodes),axis=1)



give_probabilities_to_edges(df_edges_undirected,df_nodes)

fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=df_nodes['adjusted accidents/aadb']*1000000,edge_linewidth=0,edge_color=ec,node_alpha = 1,node_color='w', bgcolor='k',use_geom=True, axis_off=False,show=False, close=False)

fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges_undirected['probability']*70000,use_geom=True, axis_off=False,show=False, close=False)


fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges_undirected['length']*.001,use_geom=True, axis_off=False,show=False, close=False)


f = df_edges_undirected['length'].mean() / df_edges_undirected['probability'].mean()
df_edges_undirected['balanced_weight'] = df_edges_undirected.apply(lambda x: x['length'] + x['probability']*f,axis=1)
df_edges_undirected[['length','probability','balanced_weight']].head(20)
fig,ax = ox.plot_graph(G_undirected_with_traffic_weights, node_zorder=2,node_size=0.00,node_alpha = 0.1,node_color='k', bgcolor='k', edge_linewidth=df_edges_undirected['balanced_weight']*.001,use_geom=True, axis_off=False,show=False, close=False)
#df_edges['length'].div(df_edges['balanced_weight']).mean()

df_edges_final_one_way = ox.graph_to_gdfs(DataLoader.load_graph(name,north,south,east,west),nodes=False)
df_edges_final_one_way_with_weights = pd.merge(df_edges_final_one_way,df_edges_undirected[['u','v','probability','balanced_weight']],how='left',on=['u','v'])
df_edges_swapped = df_edges_undirected.copy()
df_edges_swapped['t'] = df_edges_swapped['u']
df_edges_swapped['u'] = df_edges_swapped['v']
df_edges_swapped['v'] = df_edges_swapped['t']
df_edges_final_one_way_with_weights = pd.merge(df_edges_final_one_way_with_weights,df_edges_swapped[['u','v','probability','balanced_weight']],how='left',on=['u','v'])
df_edges_final_one_way_with_weights['probability'] = df_edges_final_one_way_with_weights.fillna(0)[['probability_x','probability_y']].max(axis=1)
df_edges_final_one_way_with_weights['balanced_weight'] = df_edges_final_one_way_with_weights.fillna(0)[['balanced_weight_x','balanced_weight_y']].max(axis=1)

df_edges_final_one_way_with_weights = df_edges_final_one_way_with_weights[['u','v','key','osmid','length','geometry','probability','balanced_weight']]
df_edges_final_one_way_with_weights.head(1)
final_G = ox.graph_from_gdfs(df_nodes,df_edges_final_one_way_with_weights)
nx.is_strongly_connected(final_G)
len(final_G)
nx.number_strongly_connected_components(final_G)
len(max(nx.strongly_connected_components(final_G), key=len))
final_G = final_G.subgraph(max(nx.strongly_connected_components(final_G), key=len))
df_nodes = ox.graph_to_gdfs(final_G,edges=False)
df_edges_final_one_way_with_weights = ox.graph_to_gdfs(final_G,nodes=False)
final_G = ox.graph_from_gdfs(df_nodes,df_edges_final_one_way_with_weights)

orig = ox.get_nearest_node(final_G,(south+(north-south)*.5,west+(east-west)*.7))
dest = ox.get_nearest_node(final_G,(south+(north-south)*.2,west+(east-west)*.7))
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

df_nodes.to_pickle('./nodes.pkl',protocol = 4)
df_edges_final_one_way_with_weights.to_pickle('./edges.pkl',protocol = 4)
nx.write_gpickle(final_G,'./graph.pkl',protocol = 4)


df_traffic_test = df_traffic_grouped_with_features[df_traffic_grouped_with_features['setyear'] == 2016]
df_accidents_test = df_bike_accidents_in_region[df_bike_accidents_in_region['CRASH_YEAR'] == 2016]

df_traffic_test.plot.scatter(x='x', y='y')
df_accidents_test.plot.scatter(x='DEC_LONG', y='DEC_LAT')


df_bike_accidents_in_region.head()

len(df_traffic_test)
len(df_accidents_test)

df_nodes_copy = df_nodes.copy()
df_edges_final_one_way_with_weights_copy = df_edges_final_one_way_with_weights.copy()


Tagger.number_of_crashes_at_a_node(df_accidents_test,df_nodes_copy,prefix = 'test_')
df_nodes_copy['test_number_of_accidents'].sum()
df_traffic_test['accidents_on_edge'] = df_traffic_test.apply(lambda r: df_nodes_copy.loc[r['u']]['test_number_of_accidents'] + df_nodes_copy.loc[r['v']]['test_number_of_accidents'],axis=1)
df_traffic_test['accidents_on_edge'].sum()
df_traffic_test['accidents_on_edge'].describe()
df_traffic_test.head()


df_traffic_test['empirical_ratio'] = df_traffic_test['accidents_on_edge'].div(365*df_traffic_test['aadb'])
df_traffic_test['empirical_ratio'].mean()
final_eval = pd.merge(df_traffic_test,ox.graph_to_gdfs(final_G.to_undirected(),nodes=False),how='left',on = ['u','v','key'])[['u','v','key','probability','empirical_ratio']]


final_eval.plot.scatter(x='probability',y='empirical_ratio')

final_eval['probability_scaled'] = final_eval['probability']/final_eval['probability'].mean()
final_eval['emp_scaled'] = final_eval['empirical_ratio']/final_eval['empirical_ratio'].mean()

final_eval.cov()

1.011234e-12 / np.sqrt(1.397332e-12)/np.sqrt(7.781007e-11)


final_eval['emp_scaled'].mean()

final_eval.cov()

final_eval['probability'].sum()
final_eval['empirical_ratio'].sum()
final_eval['empirical_ratio'][0:20]
(final_eval['empirical_ratio']-final_eval['empirical_ratio'].mean())[0:40]
(final_eval['empirical_ratio']-final_eval['probability'])[0:40]

final_eval['empirical_ratio'].mean()

final_eval['empirical_ratio'].mean()

np.abs((final_eval['empirical_ratio']-0)).sum()
np.abs((final_eval['empirical_ratio']-final_eval['probability'])).sum()

np.sqrt(np.square((final_eval['empirical_ratio']-final_eval['empirical_ratio'].mean())).sum())
np.sqrt(np.square((final_eval['empirical_ratio']-final_eval['probability'])).sum())

np.sqrt(np.square((final_eval['emp_scaled']-final_eval['emp_scaled'].mean())).sum())
np.sqrt(np.square((final_eval['emp_scaled']-final_eval['probability_scaled'])).sum())


final_eval[['empirical_ratio','probability']][0:40]

final_eval[0:20]

final_eval.cov()

final_eval['u_before'] = final_eval['u']
final_eval['v_before'] = final_eval['v']
final_eval['u'] = final_eval['v']
final_eval['v'] = final_eval['u']

final_eval = pd.merge(final_eval,ox.graph_to_gdfs(final_G,nodes=False),how='left',on = ['u','v','key'])

final_eval[['probability_x','probability_y','empirical_ratio']]

final_eval.head()

ox.graph_to_gdfs(final_G,nodes=False)['probability']

final_eval['probability']
alpha,beta
