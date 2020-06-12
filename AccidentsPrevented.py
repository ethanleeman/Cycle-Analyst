len(final_G)
import math
import random

prob1 = 0
prob2 = 0
prob3 = 0
len1 = 0
len2 = 0
len3 = 0

random.randrange(start=0,stop=11380)

for i in range(1000):
    orig = list(final_G)[random.randrange(start=0,stop=11380)]
    dest = list(final_G)[random.randrange(start=0,stop=11380)]
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

    prob1 += route_1_probability
    prob2 += route_2_probability
    prob3 += route_3_probability
    len1 += route_1_length
    len2 += route_2_length
    len3 += route_3_length
print(prob1)
print(prob2)
print(prob3)
print(len1)
print(len2)
print(len3)
print()
print(prob1/prob1)
print(prob2/prob1)
print(prob3/prob1)
print(len1/len1)
print(len2/len1)
print(len3/len1)
