
import streamlit as st
import pydeck as pdk
import pandas as pd
import networkx as nx
import osmnx as ox
from streamlit import caching
import SessionState
#from geopy.geocoders import Nominatim
#geolocator = Nominatim(user_agent="cycle-analyst")



def get_node_df(location):
    #Inputs: location as tuple of coords (lat, lon)
    #Returns: 1-line dataframe to display an icon at that location on a map

    #Location of Map Marker icon
    icon_data = {
        "url": "https://img.icons8.com/ultraviolet/80/000000/marker.png",
        "width": 128,
        "height":128,
        "anchorY": 128}

    return pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'icon_data': [icon_data]})

def get_node_df_end(location):
    #Inputs: location as tuple of coords (lat, lon)
    #Returns: 1-line dataframe to display an icon at that location on a map

    #Location of Map Marker icon
    icon_data = {
        "url": "https://img.icons8.com/office/80/000000/marker.png",
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

    gdf_nodes = pd.read_pickle('./data/app/nodes.pkl')
    gdf_edges = pd.read_pickle('./data/app/edges.pkl')
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
def nodes_to_lats_lons(nodes, path_nodes,offset = 0):
    #Inputs: node df, and list of nodes along path
    #Returns: 4 lists of source and destination lats/lons for each step of that path for LineLayer
    #S-lon1,S-lat1 -> S-lon2,S-lat2; S-lon2,S-lat2 -> S-lon3,S-lat3...
    source_lats = []
    source_lons = []
    dest_lats = []
    dest_lons = []

    for i in range(0,len(path_nodes)-1):
        source_lats.append(nodes.loc[path_nodes[i]]['y']+offset)
        source_lons.append(nodes.loc[path_nodes[i]]['x']+offset)
        dest_lats.append(nodes.loc[path_nodes[i+1]]['y']+offset)
        dest_lons.append(nodes.loc[path_nodes[i+1]]['x']+offset)

    return (source_lats, source_lons, dest_lats, dest_lons)


def source_to_dest(G, gdf_nodes, gdf_edges, s, e):
    #Inputs: Graph, nodes, edges, source, end, distance to walk, pace = speed, w2 bool = avoid busy roads

    if s == '':
        #No address, default to City Hall
        st.write('Source address not found, defaulting to Independence National Historical Park')
        s = 'Independence National Historical Park Philadelphia'
        start_location = ox.utils_geo.geocode(s)
    else:
        try:
            start_location = ox.utils_geo.geocode(s + ' Philadelphia')
        except:
            #No address found, default to City Hall
            st.write('Source address not found, defaulting to Independence National Historical Park')
            s = 'Independence National Historical Park Philadelphia'
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

    if start_location == end_location:
        st.write('Source and Destination identical.')
        st.write('Defaulting to Independence National Historical Park Philadelphia to University of Pennsylvania')
        s = 'Independence National Historical Park Philadelphia'
        start_location = ox.utils_geo.geocode(s)
        e = '3401 Walnut St Philadelphia'
        end_location = ox.utils_geo.geocode(e)


    #Get coordinates from addresses
    start_coords = (start_location[0], start_location[1])
    end_coords = (end_location[0], end_location[1])

    #Snap addresses to graph nodes
    start_node = ox.get_nearest_node(G, start_coords)
    end_node = ox.get_nearest_node(G, end_coords)




    shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
    short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
    short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
    short_layer = make_linelayer(short_df, '[200,000,000]')


    offset = .001

    safe_route = nx.shortest_path(G, start_node, end_node, weight = 'probability')
    safe_start_lat, safe_start_lon, safe_dest_lat, safe_dest_lon = nodes_to_lats_lons(gdf_nodes, safe_route,offset=0.00005)
    safe_df = pd.DataFrame({'startlat':safe_start_lat, 'startlon':safe_start_lon, 'destlat': safe_dest_lat, 'destlon':safe_dest_lon})
    safe_layer = make_linelayer(safe_df, '[000,200,0]')

    balanced_route = nx.shortest_path(G, start_node, end_node, weight = 'balanced_weight')
    balanced_start_lat, balanced_start_lon, balanced_dest_lat, balanced_dest_lon = nodes_to_lats_lons(gdf_nodes, balanced_route,offset=-0.00005)
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
    end_node_df = get_node_df_end(end_location)
    icon_layer = make_iconlayer(start_node_df)
    icon_layer_end = make_iconlayer(end_node_df)
    optimized_layer = make_linelayer(opt_df, '[50,220,50]')

    accident_layer = make_accidentlayer(gdf_nodes[gdf_nodes['number_of_accidents']>0][['accidents_scaled','x','y']],'[0,250,250]')


    pdk_ret = pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13, max_zoom = 15, min_zoom = 12),layers=[short_layer,safe_layer,balanced_layer, icon_layer,icon_layer_end,accident_layer])




    # st.pydeck_chart(pdk.Deck(
    #     initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13, max_zoom = 15, min_zoom = 12),
    #     layers=[short_layer,safe_layer,balanced_layer,accident_layer, icon_layer]))
    #
    route_1_length      = sum(ox.utils_graph.get_route_edge_attributes(G,shortest_route,'length'))
    route_1_probability = sum(ox.utils_graph.get_route_edge_attributes(G,shortest_route,'probability'))
    route_2_length      = sum(ox.utils_graph.get_route_edge_attributes(G,safe_route,'length'))
    route_2_probability = sum(ox.utils_graph.get_route_edge_attributes(G,safe_route,'probability'))
    route_3_length      = sum(ox.utils_graph.get_route_edge_attributes(G,balanced_route,'length'))
    route_3_probability = sum(ox.utils_graph.get_route_edge_attributes(G,balanced_route,'probability'))
    #
    df = pd.DataFrame({'A' : ['Color','Distance (meters)','Decrease of accidents'],
                        'Fastest Route':['Red',int(route_1_length),'No decrease (0%)'],
                        'Safest Route':['Green',int(route_2_length),str(int((1-route_2_probability/route_1_probability)*100)) +'% reduction'],
                        'Safe/Fast Balance':['Yellow',int(route_3_length),str(int((1-route_3_probability/route_1_probability)*100)) +'% reduction']})
    df = df.set_index('A')
    # st.write(df)
    # st.write('Bubbles on map: How many accidents have happened at this intersection in the last 20 years. Green and Yellow routes will try to avoid these \'accident hotspots\'.')


    # st.write('The red is the fastest, green is the safest, yellow is a balance.')
    # st.write('Short Route (red)  length     : ' + str(int(route_1_length)) + ' meters')
    # st.write('Short Route (red)  prob       : once in ' + str(int(1/route_1_probability)) +' rides')
    # st.write('Safe Route (green) length     : ' + str(int(route_2_length))  + ' meters')
    # st.write('Safe Route (green) prob       : once in ' + str(int(1/route_2_probability))+' rides')
    # st.write('Balanced Route (yellow) length: ' + str(int(route_3_length))  + ' meters')
    # st.write('Balanced Route (yellow) prob  : once in ' + str(int(1/route_3_probability))+' rides')



    return pdk_ret, df
#
# ############################################################################
@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_graph_from_pickle():
    return nx.read_gpickle('./data/app/graph.pkl')



with st.spinner('Making Graph...'):
    gdf_nodes, gdf_edges = get_gdfs()
    gdf_nodes['accidents_scaled'] = gdf_nodes['number_of_accidents']*5
with st.spinner('Building Local Network...'):
    G = get_graph_from_pickle()
    #G = get_graph(gdf_nodes,gdf_edges)
accident_layer = make_accidentlayer(gdf_nodes[gdf_nodes['number_of_accidents']>0][['accidents_scaled','x','y']],'[0,250,250]')


pdk_output = pdk.Deck(initial_view_state=pdk.ViewState(latitude = 39.9526, longitude = -75.1652, zoom=11),layers=[accident_layer])
df_output = pd.DataFrame({'A' : ['Color','Distance (meters)','Decrease of accidents'],
                    'Fastest Route':['Red','',' '],
                    'Safest Route':['Green',' ',' '],
                    'Safe/Fast Balance':['Yellow',' ',' ']})
df_output = df_output.set_index('A')


state = SessionState.get(pdk = pdk_output, df=df_output)


st.sidebar.markdown("## Welcome to the Cycle-Analyst!\n\n Enter an origin and destination, or press 'Calculate Route' for a default pair of locations. \n\n Some example inputs: \n\n - Points of Interest \n   - The Liberty Bell \n   - The Philadelphia Museum of Art \n - Neighborhoods \n     - Spring Garden \n   - Chinatown \n \n - Addresses \n    - 2000 Fitzwater St\n    - 1350 Walnut St")
st.sidebar.markdown("Blue dots correspond to accidents that have occured at the intersection in the last 20 years. A larger dot means more accidents.")
st.sidebar.markdown("The balanced route weighs both time and safety in equal amounts, often giving the best of both worlds! Compare all three routes in the table below the map.")
st.sidebar.markdown("## Happy (and safe) cycling!")

# #Main
st.header("Cycle-Analyst of Downtown Philadelphia")
st.header("")
st.markdown('Plan your bike ride:')

input1 = st.text_input('Input Start of Bike Ride:')
input2 = st.text_input('Input End of Bike Ride:')

#submit = st.button('Calculate route - Go!', key=1)
if st.button('Calculate route'):
    with st.spinner('Routing...'):
        state.pdk,state.df = source_to_dest(G, gdf_nodes, gdf_edges, input1, input2)


st.pydeck_chart(state.pdk)
st.write(state.df)



# gdf_nodes = pd.read_pickle('./nodes.pkl')
# gdf_edges = pd.read_pickle('./edges.pkl')
# G = ox.graph_from_gdfs(gdf_nodes,gdf_edges)
# nx.write_gpickle(G,'./graph.pkl',protocol = 4)

# if not submit:
#     st.pydeck_chart(pdk.Deck(
#         initial_view_state=pdk.ViewState(latitude = 39.9526, longitude = -75.1652, zoom=11)))
# else:
#     with st.spinner('Making Graph...'):
#         gdf_nodes, gdf_edges = get_gdfs()
#     with st.spinner('2'):
#         gdf_nodes['accidents_scaled'] = gdf_nodes['number_of_accidents']*5
#     with st.spinner('Building Local Network...'):
#         G = get_graph(gdf_nodes,gdf_edges)
#     with st.spinner('Routing...'):
#         source_to_dest(G, gdf_nodes, gdf_edges, input1, input2)
