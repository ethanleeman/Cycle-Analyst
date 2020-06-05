import pickle
import pandas as pd
import os
import networkx as nx
import overpy
import osmnx as ox
import shapely.geometry as geometry
from shapely.ops import linemerge, unary_union, polygonize


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
    return geometry.MultiPolygon(polygons)

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
