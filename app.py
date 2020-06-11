
import streamlit as st
import pydeck as pdk
import pandas as pd
import networkx as nx
import osmnx as ox
from streamlit import caching

#
def get_node_df(location):
    #Inputs: location as tuple of coords (lat, lon)
    #Returns: 1-line dataframe to display an icon at that location on a map

    #Location of Map Marker icon
    icon_data = {
        "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
        "width": 128,
        "height":128,
        "anchorY": 128}

    return pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'icon_data': [icon_data]})

# def get_text_df(text, location):
#     #Inputs: text to display and location as tuple of coords (lat, lon)
#     #Returns: 1-line dataframe to display text at that location on a map
#     return pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'text':text})
#
# ############################################################################
#
def make_iconlayer(df):
    #Inputs: df with [lat, lon, icon_data]
    #Returns: pydeck IconLayer
    return pdk.Layer(
       type='IconLayer',
       data=df,
       get_icon='icon_data',
       opacity=0.6,
        get_size=4,
        pickable=True,
        size_scale=15,
        get_position='[lon, lat]')

# def make_textlayer(df, color_array):
#     #Inputs: df with [lat, lon, text] and font color as str([R,G,B]) - yes '[R,G,B]'
#     #Returns: pydeck TextLayer
#     return pdk.Layer(
#         type='TextLayer',
#         data=df,
#         get_text='text',
#         get_size=4,
#         pickable=True,
#         size_scale=6,
#         getColor = color_array,
#         get_position='[lon, lat]')
#



def make_accidentlayer(df, color_array):
        return pdk.Layer(
            type='ScatterplotLayer',
            data=df,
            opacity=0.1,
            get_position = ['x','y'],
            get_fill_color = color_array,
            get_radius='accidents_scaled')

def make_linelayer(df, color_array):
    #Inputs: df with [startlat, startlon, destlat, destlon] and font color as str([R,G,B]) - yes '[R,G,B]'
    #Plots lines between each line's [startlon, startlat] and [destlon, destlat]
    #Returns: pydeck LineLayer
    return pdk.Layer(
        type='LineLayer',
        data=df,
        opacity=0.1,
        getSourcePosition = '[startlon, startlat]',
        getTargetPosition = '[destlon, destlat]',
        getColor = color_array,
        getWidth = '5')
#
# ############################################################################
#
@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_map():
    #Returns: map as graph from graphml
    #Cached by Streamlit

    G = ox.load_graphml(filepath='./philadelphia.graphml')
    return G
#
@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_gdfs():
    #Returns: nodes and edges from pickle
    #Cached by Streamlit

    gdf_nodes = pd.read_pickle('./nodes.pkl')
    gdf_edges = pd.read_pickle('./edges.pkl')
    return gdf_nodes, gdf_edges

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_graph(gdf_nodes,gdf_edges):
    #Returns: nodes and edges from pickle
    #Cached by Streamlit

    return ox.graph_from_gdfs(gdf_nodes,gdf_edges)

#
# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
# def set_walking_rate(rate):
#     #Inputs: walking rate
#     #Returns: walking rate (cached for streamlit)
#     return int(rate)
#
# ############################################################################
#
# def get_dist(lat1, lon1, lat2, lon2):
#     #Inputs: 4 integers, latitudes and longitudes from point 1 followed by point 2
#     #Returns: birds-eye distance between them
#
#     #Coefficients are distance across 1degree Lat/Long
#     #Precalculated for Boston lat and lon
#     return (111073*abs(lat1-lat2)+82850*abs(lon1-lon2))
#
def get_map_bounds(gdf_nodes, route1, route2,route3):
    #Inputs: node df, and two lists of nodes along path
    #Returns: Coordinates of smallest rectangle that contains all nodes
    max_x = -1000
    min_x = 1000
    max_y = -1000
    min_y = 1000

    for i in (route1 + route2 + route3):
        row = gdf_nodes.loc[i]
        temp_x = row['x']
        temp_y = row['y']

        max_x = max(temp_x, max_x)
        min_x = min(temp_x, min_x)
        max_y = max(temp_y, max_y)
        min_y = min(temp_y, min_y)

    return min_x, max_x, min_y, max_y
#
def nodes_to_lats_lons(nodes, path_nodes):
    #Inputs: node df, and list of nodes along path
    #Returns: 4 lists of source and destination lats/lons for each step of that path for LineLayer
    #S-lon1,S-lat1 -> S-lon2,S-lat2; S-lon2,S-lat2 -> S-lon3,S-lat3...
    source_lats = []
    source_lons = []
    dest_lats = []
    dest_lons = []

    for i in range(0,len(path_nodes)-1):
        source_lats.append(nodes.loc[path_nodes[i]]['y'])
        source_lons.append(nodes.loc[path_nodes[i]]['x'])
        dest_lats.append(nodes.loc[path_nodes[i+1]]['y'])
        dest_lons.append(nodes.loc[path_nodes[i+1]]['x'])

    return (source_lats, source_lons, dest_lats, dest_lons)
#
# ############################################################################
#
# def beautreeful_node(G, node_list):
#     #Inputs: G, list of nodes
#     #Returns: list of nodes rich in trees to route to
#     best_node = node_list[0]
#     best_node_score = 0
#     best_neighbors = []
#
#     tree_df = pd.read_csv('data/combined_nodetrees.csv')
#
#     #For each promising node
#     for i in node_list:
#         #If that node has trees
#         if i in tree_df.node.values:
#             #Keep track of tree score
#             #Initialize to trees nearest that node
#             tree_score = tree_df[tree_df.node==i].trees.tolist()[0]
#
#             #Find tree counts of that node's neighbors
#             neighbors = G.neighbors(i)
#             neighbor_ids = []
#             neighbor_scores = []
#             for neighbor in neighbors:
#                 #If the neighboring node has trees
#                 if neighbor in tree_df.node.values:
#                     #Keep track of its trees and ID
#                     neighbor_trees = tree_df[tree_df.node==neighbor].trees.tolist()[0]
#                     neighbor_scores.append(neighbor_trees)
#                     neighbor_ids.append(neighbor)
#
#                 #Place neighbor IDs in order of their tree scores
#                 neighbor_ids = [x for _,x in sorted(zip(neighbor_scores, neighbor_ids), reverse=True)]
#
#             if len(neighbor_ids) > 0:
#                 #We include neighbor's tree counts into this calculation
#                 tree_score += int(sum(neighbor_scores)/max(1,len(neighbor_ids)))
#
#             if tree_score > best_node_score:
#                 best_neighbors = neighbor_ids[:]
#                 best_node_score = tree_score
#                 best_node = i
#
#     if len(best_neighbors) < 1:
#         return [best_node]
#     elif len(best_neighbors) == 1:
#         return [best_node, best_neighbors[0]]
#     else:
#         return [best_neighbors[0], best_node, best_neighbors[1]]
#
# def find_midpoint(G, start_node, dist):
#     #Inputs: G, index of start_node, distance of walk
#     #Returns: index of midpoint_node, list of indices of nodes along path to midpoint
#
#     #Step 1: Identify contender midpoints within factor1*dist
#     factor1 = 0.5
#     contender_midpoints = nx.single_source_dijkstra(G, start_node, weight = 'length', cutoff = factor1*dist)
#     #Returns 2 dicts: first of end nodes:distance and second of end node:node path
#
#     #Dict [node index]:distance from start node
#     contender_paths = contender_midpoints[0]
#
#     #All contender nodes that are within factor2 of target length
#     factor2 = 0.9
#     farthest_node_considered = max(contender_paths.values())
#     midpoint_nodes = [k for (k, v) in contender_paths.items() if v >= factor2*farthest_node_considered]
#
#     #Step 2: Find contender midpoint node in a tree-rich area
#     midpoints = beautreeful_node(G, midpoint_nodes)
#
#     return midpoints
#
# def find_park(G, parks, start_node, start_location, dist):
#     #Inputs: G, parks df, index of start_node, coords of start_node, distance of walk
#     #Returns: index of midpoint_node index of chosen park
#
#     #Find a park close to 1/2 distance
#     dists = []
#     for row in parks.itertuples():
#         dists.append(get_dist(getattr(row,'lat'), getattr(row,'lon'), start_location[0], start_location[1]))
#
#     #Find a park closest to factor*the distance you'd like to travel
#     factor = 0.5
#     dists = [abs(x-factor*dist) for x in dists]
#
#     #Identify node closest to coordinates of chosen park
#     index = dists.index(min(dists))
#     midpoint_coords = (parks.iloc[index]['lat'], parks.iloc[index]['lon'])
#     midpoint = ox.get_nearest_node(G, midpoint_coords)
#
#     return midpoint, index
#
# def source_to_source(G, gdf_nodes, gdf_edges, s, dist, to_park, avoid_streets):
#     #Inputs: Graph, nodes, edges, source, distance to walk, to_park bool = route to park, avoid_streets bool = avoid busy roads
#
#     #Set default source to Insight Offices
#     if s == '':
#         #No address, default to Insight
#         st.write('Source address not found, defaulting to Insight Offices')
#         s = '280 Summer St Boston'
#         start_location = ox.geo_utils.geocode(s)
#     else:
#         try:
#             start_location = ox.geo_utils.geocode(s + ' Boston')
#         except:
#             #No address found, default to Insight
#             st.write('Source address not found, defaulting to Insight Offices')
#             s = '280 Summer St Boston'
#             start_location = ox.geo_utils.geocode(s)
#
#     #Get coordinates from start address and snap to node
#     start_coords = (start_location[0], start_location[1])
#     start_node = ox.get_nearest_node(G, start_coords)
#
#     #Calculate new edge weights
#     tree_counts = {}
#     road_safety = {}
#     lengths = {}
#
#     for row in gdf_edges.itertuples():
#         u = getattr(row,'u')
#         v = getattr(row,'v')
#         key = getattr(row, 'key')
#         tree_count = getattr(row, 'trees')
#         safety_score = getattr(row, 'safety')
#         length = getattr(row, 'length')
#
#         tree_counts[(u,v,key)] = tree_count
#         road_safety[(u,v,key)] = safety_score
#         lengths[(u,v,key)] = length
#
#     #Optimized attribute is a weighted combo of normal length, tree counts, and road safety.
#     #Larger value is worse
#     optimized = {}
#     for key in lengths.keys():
#         temp = int(lengths[key])
#         temp += int(250/int(max(1,tree_counts[key])))
#         if avoid_streets:
#             temp += int(100*(road_safety[key]))
#         optimized[key] = temp
#
#     #Generate new edge attributes - depending on user prefs
#     nx.set_edge_attributes(G, optimized, 'optimized')
#
#     if to_park:
#         #Get coords of park and path to it
#         parks = pd.read_csv('data/parks.csv')
#         midpoint, park_index = find_park(G, parks, start_node, start_location, dist)
#
#         #Get path to park
#         path = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')
#
#         #Display name of park on the map
#         text_layer = make_textlayer(get_text_df(parks.iloc[park_index]['Name'], (parks.iloc[park_index]['lat'], parks.iloc[park_index]['lon'])), '[0,0,0]')
#
#     else:
#         #Get coords of midpoint nodes
#         midpoints = find_midpoint(G, start_node, dist)
#
#         #Get path to midpoint
#         path = nx.shortest_path(G, start_node, midpoints[0], weight = 'optimized')
#
#         if len(midpoints) > 1:
#             path.append(midpoints[1])
#         if len(midpoints) > 2:
#             path.append(midpoints[2])
#
#         #Dummy code because we are not outputting text
#         text_layer = make_textlayer(get_text_df('', start_coords), '[255,255,255]')
#
#     #Step 3: Adjust edge weights to penalize backtracking
#     for node in range(-1+len(path)):
#         G[path[node]][path[node+1]][0]['optimized'] += 300
#
#     #Step 4: Get new route back
#     if to_park:
#         route_back = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')
#     else:
#         route_back = nx.shortest_path(G, start_node, midpoints[len(midpoints)-1], weight = 'optimized')
#
#     #Step 5: Reset edge weights
#     for node in range(-1+len(path)):
#         G[path[node]][path[node+1]][0]['optimized'] -= 300
#
#     #Step 6: Plot route
#     loop1_start_lat, loop1_start_lon, loop1_dest_lat, loop1_dest_lon = nodes_to_lats_lons(gdf_nodes, path)
#     loop2_start_lat, loop2_start_lon, loop2_dest_lat, loop2_dest_lon = nodes_to_lats_lons(gdf_nodes, route_back)
#
#     #This finds the bounds of the final map to show based on the paths
#     min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, path, route_back)
#
#     #Find the average lat/long to center the map
#     center_x = 0.5*(max_x + min_x)
#     center_y = 0.5*(max_y + min_y)
#
#     #Move coordinates into dfs
#     loop1_df = pd.DataFrame({'startlat':loop1_start_lat, 'startlon':loop1_start_lon, 'destlat': loop1_dest_lat, 'destlon':loop1_dest_lon})
#     loop2_df = pd.DataFrame({'startlat':loop2_start_lat, 'startlon':loop2_start_lon, 'destlat': loop2_dest_lat, 'destlon':loop2_dest_lon})
#
#     #Add map marker icon at start
#     start_node_df = get_node_df(start_location)
#     outbound_layer = make_linelayer(loop1_df, '[150,150,220]')
#     inbound_layer = make_linelayer(loop2_df, '[220,50,50]')
#     icon_layer = make_iconlayer(start_node_df)
#
#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom = 13, max_zoom = 15, min_zoom = 12),
#         layers=[text_layer, outbound_layer, inbound_layer, icon_layer]))
#
#     st.write('From your location, take the blue path to the turnaround point. Then return via the red path.')
#     return
#
# ############################################################################
#
def source_to_dest(G, gdf_nodes, gdf_edges, s, e):
    #Inputs: Graph, nodes, edges, source, end, distance to walk, pace = speed, w2 bool = avoid busy roads

    if s == '':
        #No address, default to City Hall
        st.write('Source address not found, defaulting to City Hall Station')
        s = 'City Hall Station Philadelphia'
        start_location = ox.utils_geo.geocode(s)
    else:
        try:
            start_location = ox.utils_geo.geocode(s + ' Philadelphia')
        except:
            #No address found, default to City Hall
            st.write('Source address not found, defaulting to City Hall')
            s = '1400 John F Kennedy Blvd Philadelphia'
            start_location = ox.utils_geo.geocode(s)

    if e == '':
        #No address, default to UPenn
        st.write('Destination address not found, defaulting to University of Pennsylvania')
        e = '3401 Walnut St Philadelphia'
        end_location = ox.utils_geo.geocode(e)
    else:
        try:
            end_location = ox.utils_geo.geocode(e + ' Philadelphia')
        except:
            #No address found, default to Insight
            st.write('Destination address not found, defaulting to University of Pennsylvania')
            e = '3401 Walnut St Philadelphia'
            end_location = ox.utils_geo.geocode(e)

    #Get coordinates from addresses
    start_coords = (start_location[0], start_location[1])
    end_coords = (end_location[0], end_location[1])

    #Snap addresses to graph nodes
    start_node = ox.get_nearest_node(G, start_coords)
    end_node = ox.get_nearest_node(G, end_coords)




    #
    # #Calculate new edge weights
    # tree_counts = {}
    # road_safety = {}
    # lengths = {}
    #
    # for row in gdf_edges.itertuples():
    #     u = getattr(row,'u')
    #     v = getattr(row,'v')
    #     key = getattr(row, 'key')
    #     tree_count = getattr(row, 'trees')
    #     safety_score = getattr(row, 'safety')
    #     length = getattr(row, 'length')
    #
    #     tree_counts[(u,v,key)] = tree_count
    #     road_safety[(u,v,key)] = safety_score
    #     lengths[(u,v,key)] = length
    #
    # #We need to make sure that dist is at least the length of the shortest path
    # min_dist = nx.shortest_path_length(G, start_node, end_node, weight = 'length')
    #
    # if dist < min_dist:
    #     st.write('This walk will probably take a bit longer - approximately ' + str(int(min_dist/pace)) + ' minutes total.')
    #
    #     #We are in a rush, take the shortest path
    #     optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
    #
    # #Optimized attribute is a weighted combo of normal length, tree counts, and road safety.
    # #Larger value is worse
    # elif dist < 1.3*min_dist:
    #     #We have some extra time, length term still important
    #     optimized = {}
    #     for key in lengths.keys():
    #         temp = int(lengths[key])
    #         temp += int(250/int(max(1,tree_counts[key])))
    #         if avoid_streets:
    #             temp += int(100*(road_safety[key]))
    #         optimized[key] = temp
    #
    #     #Generate new edge attribute
    #     nx.set_edge_attributes(G, optimized, 'optimized')
    #
    #     #Path of nodes
    #     optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')
    #
    # else:
    #     #dist > 1.3*min_dist
    #     opt_dist = nx.shortest_path_length(G, start_node, end_node, weight = 'optimized')
    #     if dist > 1.3*opt_dist:
    #         st.write('You can take your time! The walk should only take approximately ' + str(int(opt_dist/pace)) + ' minutes.')
    #     #We have a lot of extra time, let trees/safety terms dominate
    #     optimized = {}
    #     for key in lengths.keys():
    #         temp = 0.2*int(lengths[key])
    #         temp += int(250/int(max(1,tree_counts[key])))
    #         if avoid_streets:
    #             temp += int(100*(road_safety[key]))
    #         optimized[key] = temp
    #
    #     #Generate new edge attributes
    #     nx.set_edge_attributes(G, optimized, 'optimized')
    #
    #     #Path of nodes
    #     optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')

    shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
    short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
    short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
    short_layer = make_linelayer(short_df, '[200,000,000]')


    safe_route = nx.shortest_path(G, start_node, end_node, weight = 'probability')
    safe_start_lat, safe_start_lon, safe_dest_lat, safe_dest_lon = nodes_to_lats_lons(gdf_nodes, safe_route)
    safe_df = pd.DataFrame({'startlat':safe_start_lat, 'startlon':safe_start_lon, 'destlat': safe_dest_lat, 'destlon':safe_dest_lon})
    safe_layer = make_linelayer(safe_df, '[000,200,0]')

    balanced_route = nx.shortest_path(G, start_node, end_node, weight = 'balanced_weight')
    balanced_start_lat, balanced_start_lon, balanced_dest_lat, balanced_dest_lon = nodes_to_lats_lons(gdf_nodes, balanced_route)
    balanced_df = pd.DataFrame({'startlat':balanced_start_lat, 'startlon':balanced_start_lon, 'destlat': balanced_dest_lat, 'destlon':balanced_dest_lon})
    balanced_layer = make_linelayer(balanced_df, '[200,200,00]')

    #This finds the bounds of the final map to show based on the paths
    min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, safe_route,balanced_route)

    #These are lists of origin/destination coords of the paths that the routes take
    opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)

    #Find the average lat/long to center the map
    center_x = 0.5*(max_x + min_x)
    center_y = 0.5*(max_y + min_y)

    #Move coordinates into dfs
    opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

    start_node_df = get_node_df(start_location)
    icon_layer = make_iconlayer(start_node_df)
    optimized_layer = make_linelayer(opt_df, '[50,220,50]')

    accident_layer = make_accidentlayer(gdf_nodes[gdf_nodes['number_of_accidents']>0][['accidents_scaled','x','y']],'[0,250,250]')

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13, max_zoom = 15, min_zoom = 12),
        layers=[short_layer,safe_layer,balanced_layer,accident_layer, icon_layer]))

    route_1_length      = sum(ox.utils_graph.get_route_edge_attributes(G,shortest_route,'length'))
    route_1_probability = sum(ox.utils_graph.get_route_edge_attributes(G,shortest_route,'probability'))
    route_2_length      = sum(ox.utils_graph.get_route_edge_attributes(G,safe_route,'length'))
    route_2_probability = sum(ox.utils_graph.get_route_edge_attributes(G,safe_route,'probability'))
    route_3_length      = sum(ox.utils_graph.get_route_edge_attributes(G,balanced_route,'length'))
    route_3_probability = sum(ox.utils_graph.get_route_edge_attributes(G,balanced_route,'probability'))
    st.write('The red is the fastest, green is the safest, yellow is a balance.')
    st.write('Short Route (red)  length     : ' + str(int(route_1_length)) + ' meters')
    st.write('Short Route (red)  prob       : once in ' + str(int(1/route_1_probability)) +' rides')
    st.write('Safe Route (green) length     : ' + str(int(route_2_length))  + ' meters')
    st.write('Safe Route (green) prob       : once in ' + str(int(1/route_2_probability))+' rides')
    st.write('Balanced Route (yellow) length: ' + str(int(route_3_length))  + ' meters')
    st.write('Balanced Route (yellow) prob  : once in ' + str(int(1/route_3_probability))+' rides')



    return
#
# ############################################################################


gdf_nodes, gdf_edges = get_gdfs()
gdf_nodes['accidents_scaled'] = gdf_nodes['number_of_accidents']*5

G = get_graph(gdf_nodes,gdf_edges)

# #Main
st.header("Cycle-Analyst of Downtown Philadelphia")
st.header("")
st.markdown('Plan your bike ride:')

input1 = st.text_input('Input Start of Bike Ride:')
input2 = st.text_input('Input End of Bike Ride:')
#
submit = st.button('Calculate route - Go!', key=1)
if not submit:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude = 39.9526, longitude = -75.1652, zoom=11)))
else:
    with st.spinner('Routing...'):
        source_to_dest(G, gdf_nodes, gdf_edges, input1, input2)
