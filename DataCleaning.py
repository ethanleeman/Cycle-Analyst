import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import geopandas as gpd
import logging as lg
import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from osmnx import utils
from osmnx import utils_graph





def clean_crash_data(df):

    df = df[df['BICYCLE_COUNT'] > 0]
    df = df[df['DEC_LAT'] < 40.3]
    df = df[df['BICYCLE_COUNT'] > 0]
    #df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].str.split("\\s+|\.\\s*|\\:\\s*").str.join(""), downcast="float") / 10000000
    #df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].str.split("\\s+|\.\\s*|\\:\\s*").str.join(""), downcast="float") / 10000000
    return df

def clean_traffic(df):
    an = df[df['X'] < df_crash['DEC_LONG'].max()]
    an = an[an['X'] > df_crash['DEC_LONG'].min()]
    an = an[an['Y'] < df_crash['DEC_LAT'].max()]
    an = an[an['Y'] > df_crash['DEC_LAT'].min()]
    in_philly = an.apply(lambda r: philly.contains(geometry.Point(r['X'], r['Y'])),axis=1)
    an = an[in_philly]
    return an

def exploder_one_hot(df,column_name_list):
    to_concat = [df]
    for col in column_name_list:
        to_concat.append(df[col].apply(lambda x: [x]).str.join('|').str.get_dummies().add_prefix(col + ":"))
        #return df[col].apply(lambda x: [x]).str.join('|').str.get_dummies()
    return pd.concat(to_concat,axis=1).drop(column_name_list,axis=1)

def graph_to_


def _consolidate_intersections_rebuild_graph(G, tolerance=10, update_edge_lengths=True):
    """
    Consolidate intersections comprising clusters of nearby nodes.
    Merge nodes and return a rebuilt graph with consolidated intersections and
    reconnected edge geometries.
    The tolerance argument should be adjusted to approximately match street
    design standards in the specific street network, and you should always use
    a projected graph to work in meaningful and consistent units like meters.
    Returned graph's node IDs represent clusters rather than OSMIDs. Refer to
    nodes' osmid attributes for original OSMIDs. If multiple nodes were merged
    together, the osmid attribute is a list of merged nodes' osmids.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        a projected graph
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single node, with edges reconnected to this new
        node
    update_edge_lengths : bool
        if True, update the length attribute of edges reconnected to a new
        merged node; if False, just retain the original edge length
    Returns
    -------
    H : networkx.MultiDiGraph
        a rebuilt graph with consolidated intersections and reconnected
        edge geometries
    """
    # STEP 1
    # buffer nodes to passed-in distance, merge overlaps
    gdf_nodes, gdf_edges = utils_graph.graph_to_gdfs(G)
    gdf_edges = gdf_edges.set_index(["u", "v", "key"])
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make iterable to convert to GeoSeries
        buffered_nodes = [buffered_nodes]

    # STEP 2
    # attach each node to its cluster of merged nodes
    # first get the original graph's node points
    node_points = gdf_nodes[["geometry"]]

    # then turn buffered nodes into gdf and get centroids of each cluster as x, y
    node_clusters = gpd.GeoDataFrame(geometry=list(buffered_nodes), crs=node_points.crs)
    centroids = node_clusters.centroid
    node_clusters["x"] = centroids.x
    node_clusters["y"] = centroids.y

    # then spatial join to give each node the label of cluster it's within
    gdf = gpd.sjoin(node_points, node_clusters, how="left", op="within")
    gdf = gdf.drop(columns="geometry").rename(columns={"index_right": "cluster"})

    # # STEP 3
    # # if a cluster contains multiple components (i.e., it's not connected)
    # # move each component to its own cluster (otherwise you will connect
    # # nodes together that are not truly connected, e.g., nearby deadends or
    # # surface streets with bridge).
    # groups = gdf.groupby("cluster")
    # for cluster_label, nodes_subset in groups:
    #     if len(nodes_subset) > 1:
    #         # identify all the (weakly connected) component in cluster
    #         wccs = list(nx.weakly_connected_components(G.subgraph(nodes_subset.index)))
    #         if len(wccs) > 1:
    #             # if there are multiple components in this cluster
    #             suffix = 0
    #             for wcc in wccs:
    #                 # set subcluster xy to the centroid of just these nodes
    #                 subcluster_centroid = node_points.loc[wcc].unary_union.centroid
    #                 gdf.loc[wcc, "x"] = subcluster_centroid.x
    #                 gdf.loc[wcc, "y"] = subcluster_centroid.y
    #                 # move to subcluster by appending suffix to nodes cluster label
    #                 gdf.loc[wcc, "cluster"] = f"{cluster_label}-{suffix}"
    #                 suffix += 1

    # STEP 4
    # create new empty graph and copy over misc graph data
    H = nx.MultiDiGraph()
    H.graph = G.graph

    # STEP 5
    # create a new node for each cluster of merged nodes
    # regroup now that we potentially have new cluster labels from step 3
    groups = gdf.groupby("cluster")
    for cluster_label, nodes_subset in groups:

        osmids = nodes_subset.index.to_list()
        if len(osmids) == 1:
            # if cluster is a single node, add that node to new graph
            H.add_node(cluster_label, **G.nodes[osmids[0]])
        else:
            # if cluster is multiple merged nodes, create one new node to
            # represent them
            H.add_node(
                cluster_label,
                osmid=str(osmids),
                x=nodes_subset["x"].iloc[0],
                y=nodes_subset["y"].iloc[0],
            )

    # STEP 6
    # create a new edge for each edge in original graph
    # but from cluster to cluster
    for u, v, k, data in G.edges(keys=True, data=True):
        u2 = gdf.loc[u, "cluster"]
        v2 = gdf.loc[v, "cluster"]

        # only create the edge if we're not connecting the cluster
        # to itself, but always add original self-loops
        if (u2 != v2) or (u == v):
            data["u_original"] = u
            data["v_original"] = v
            if "geometry" not in data:
                data["geometry"] = gdf_edges.loc[(u, v, k), "geometry"]
            H.add_edge(u2, v2, **data)

    # STEP 7
    # for every group of merged nodes with more than 1 node in it,
    # extend the edge geometries to reach the new node point
    new_edges = utils_graph.graph_to_gdfs(H, nodes=False)
    for cluster_label, nodes_subset in groups:

        # but only if there were multiple nodes merged together,
        # otherwise it's the same old edge as in original graph
        if len(nodes_subset) > 1:

            # get coords of merged nodes point centroid to prepend or
            # append to the old edge geom's coords
            x = H.nodes[cluster_label]["x"]
            y = H.nodes[cluster_label]["y"]
            xy = [(x, y)]

            # for each edge incident to this new merged node, update
            # its geometry to extend to/from the new node's point coords
            mask = (new_edges["u"] == cluster_label) | (new_edges["v"] == cluster_label)
            for _, (u, v, k) in new_edges.loc[mask, ["u", "v", "key"]].iterrows():
                old_coords = list(H.edges[u, v, k]["geometry"].coords)
                new_coords = xy + old_coords if cluster_label == u else old_coords + xy
                new_geom = LineString(new_coords)
                H.edges[u, v, k]["geometry"] = new_geom

                # update the edge length attribute if parameterized to do so
                # otherwise just keep using the original edge length
                if update_edge_lengths:
                    H.edges[u, v, k]["length"] = new_geom.length

    return H
