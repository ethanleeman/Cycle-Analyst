import pandas as pd
import osmnx as ox
import numpy as np
from DataCleaning import clean_crash_data, exploder_one_hot, _consolidate_intersections_rebuild_graph
import importlib
importlib.reload(DataCleaning)



ox.settings.useful_tags_path = ['route','lcn','segregated','rcn','rcn_ref','lcn_ref','bridge','tunnel','oneway','lanes','highway','maxspeed','service','access','area','landuse','width','est_width','junction','cycleway:right','cycleway:left','surface','cycleway','cycleway:both']

## load graph
#G = ox.graph_from_place('Philadelphia, Pennsylvania, USA', network_type='bike',simplify=False)
G = ox.graph_from_point((39.91829,-75.18268),400,simplify=False,network_type='bike')
G.get_edge_data(5562127539,109860453)
G_proj = ox.project_graph(G)
G_consolidate = ox.consolidate_intersections(G_proj,tolerance = 10,rebuild_graph=True)
ox.plot_graph(G_consolidate, node_zorder=2,node_size=10,node_alpha =1,node_color='r', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
G_simple = ox.simplify_graph(G)
ox.plot_graph(G_simple, node_zorder=2,node_size=10,node_alpha =1,node_color='r', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)



edges= ox.graph_to_gdfs(G_simple,nodes=False)
edges.length.head()
edges.length.describe()

help(edges.geometry[3].wkb)


exploder_one_hot(edges,l).head().cycleway.unique()

ox.plot_graph(G_consolidate, node_zorder=2,node_size=10,node_alpha =1,node_color='r', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)

df_crash = clean_crash_data(df_crash)



df_crash.columns
df_crash.LOCATION_TYPE.unique()
df_crash.INTERSECT_TYPE.hist()
fig,ax = ox.plot_graph(G, node_zorder=2,node_size=0.03,node_alpha = 0.1,node_color='k', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
ax=df_crash.plot(kind='scatter',x='DEC_LONG',y='DEC_LAT',s=1,fig=fig,label='Bike Accident',ax=ax)

gdf_edge = ox.utils_graph.graph_to_gdfs(G,nodes=False)
gdf_node = ox.utils_graph.graph_to_gdfs(G,edges=False)

help(G.nodes)

gdf_edge.columns

gdf_edge.access.unique()
gdf_edge['service'].values
exploder_one_hot(gdf_edge,'highway')

gdf_edge.drop('highway', 1).join(gdf_edge['highway'].str.join('|').str.get_dummies())
gdf_edge['width'].str.join('|').str.get_dummies().columns
pd.options.display.max_rows =10000
gdf_edge['highway'][0:10000]

help(ox.graph_from_bbox)
G2 = ox.graph_from_point((39.94773,-75.1582),1000,simplify=True,network_type='bike')
ox.plot_graph(G2, node_zorder=2,node_size=10,node_alpha = 1,node_color='r', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)
ox.graph_to_gdfs(G2,nodes=False)
G2_proj = ox.project_graph(G2)
G3 = ox.consolidate_intersections(G2_proj,tolerance = 10,rebuild_graph=True)
G4 = ox.simplify_graph(G3.copy())
ox.plot_graph(G3, node_zorder=2,node_size=10,node_alpha =1,node_color='r', bgcolor='w', edge_linewidth=0.2,use_geom=True, axis_off=False,show=False, close=False)

gdf_edge_sample= ox.graph_to_gdfs(G3,nodes=False)
gdf_edge_sample.groupby(['u','v']).count()

type(G2)
G.nodes()
