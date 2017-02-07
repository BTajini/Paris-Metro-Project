# -*- encoding: utf-8 -*-

#---Author--- : Badr Tajini
#---Collaborator--- : Michael Faivre
#---Campus--- : Paris (On-Campus)
#---Release--- : 12/10/2016 - V7.0

#library implemented in the project
'-------------------------------------------------'
from __future__ import generators
import matplotlib.pyplot as plt
import re
import Queue
from pprint import pprint
import heapq
from time import time
import os
import psutil # must be installed first or deleted (library for calculating memory allocated in every current implementation)
'---------------------------------------------------'

#calculate memory allocated for every implementation  !!!!need install library psutil-4.3.1 or delete line below with import psutil!!!!
process = psutil.Process(os.getpid())#delete this line if you can't donwload and install library psutil-4.3.1
print'Memory allocated for this implementation :',(process.memory_info().rss),'\n'
'----------------------------------------------------'
#define local variable for running time of algorithm
t0 = time()
'----------------------------------------------------'

#Local variable used for project
'----------------------------------------------------'
#must edit the link to your dataset
dataset = open("/Paris-Metro-Project/Data/metro_paris.txt", "r")
list_source_destination = []
list_keys = []
my_graph = {}
my_graph_dijkstra = {}
lists_sources_lists = []
line_dataset = dataset.readlines()
'----------------------------------------------------'
#Creating our dictionary by loading our dataset (.txt)
for i in xrange(len(line_dataset) - 1): #xrange() generates numbers on demand, which is similar to "for loop" in java/c++; range() pre-computes all numbers and saves in memory, which causes the error.
    if not re.match(r'([\d]+)', line_dataset[i]):
        line_temp = i

for i in xrange(line_temp + 1, len(line_dataset) - 1):#xrange
    list_source_destination.append((int(line_dataset[i].strip().split(" ")[0]), int(line_dataset[i].strip().split(" ")[1]),
                                     float(line_dataset[i].strip().split(" ")[2])))
    list_keys.append(int(line_dataset[i].strip().split(" ")[0]))
'----------------------------------------------------'
#Graph for BFS & DFS & Shortest Path & All Paths
for k in xrange(len(list_keys)):
    for unit in list_source_destination:
        if unit[0] == k:
            lists_sources_lists.append(unit[1])
        else:
            pass
    if len(lists_sources_lists) != 0:
        my_graph[k] = lists_sources_lists
        lists_sources_lists = []
'----------------------------------------------------'
#Graph for Dijkstra Algorithm
for k in xrange(len(list_keys)):
    for unit in list_source_destination:
        if unit[0] == k:
            lists_sources_lists.append(((unit[1],unit[2])))
        else:
            pass
    if len(lists_sources_lists) != 0:
        my_graph_dijkstra[k] = lists_sources_lists
        lists_sources_lists = []
'----------------------------------------------------'
#test for implementation of graph
#print(my_graph)
#print(my_graph_dijkstra)
'----------------------------------------------------'
#call class Queue for BFS
expand_queue = Queue.Queue()
'----------------------------------------------------'
#class class Stack for DFS
class Stack():

    def __init__(self):
        self.stack = []

    def get(self):
        return self.stack.pop(0)

    def put(self, item):
        self.stack.insert(0, item)

    def empty(self):
        return len(self.stack) == 0
'----------------------------------------------------'
#ITERATIVE DEEPENING perform depth-limited DFS repeatedly, with an increasing depth limit, until a solution is found.
#iterative deepening simulates breadth-first search, but with only linear space complexity.
def iterative_search(data_structure, graph, start, end, limit=None):
    # initialize generator
    data_structure.put((graph, start, end,[]))

    while not data_structure.empty():  # topological sort
        graph, current, end, path = data_structure.get()

        # make solution depth limited
        # makes it iterative - for DFS to use all paths
        if limit and len(path) > limit:
            continue

        if current == end:
            # route done - yield result
            yield tuple(path + [current])

        if current in graph:
            # skip neighbor-less nodes
            for neighbor in graph[current]:
                # store all neighbors according to data structure
                data_structure.put(
                    (graph, neighbor, end, path + [current])
                )
'----------------------------------------------------'
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 5:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'
'----------------------------------------------------'
# dfs  - using stack
print "---Iterative DFS---"'\n'
# dfs  - using stack
gen = iterative_search(Stack(), my_graph,1, 165, limit=5)

# get only 5 first paths
dfs_path_set = set()
limit = 1
while len(dfs_path_set) < 5:
    try:
        dfs_path_set.add(next(gen))
    except StopIteration:
        limit += 1
        print "Depth limit reached, increasing depth limit to %d" % limit
        gen = iterative_search(
            Stack(), my_graph, 1, 165, limit=limit
        )

print os.linesep.join(map(str, dfs_path_set)),'\n'
'----------------------------------------------------'
#Difference between the two strategy
print "Difference BFS - DFS: %s" % str(bfs_path_set - dfs_path_set)
print "Difference DFS - BFS: %s" % str(dfs_path_set - bfs_path_set),'\n'
'----------------------------------------------------'

'''
Knowing that the shortest path will be returned first from the BFS path generator method
we can create a useful method which simply returns the shortest path found or None if no path exists.
As we are using a generator this in theory should provide similar performance
results as just breaking out and returning the first matching path in the BFS implementation.
'''
# call function of shortest paths for bfs (return the firt row
def shortest_path_bfs(q,graph, start, end):
    try:
        return next(iterative_search(q, graph, start, end))
    except StopIteration:
        return None
'----------------------------------------------------'
# call function of shortest paths for dfs ( there is some issue for increasing memory , getting memory error ) maybe lower performance of my RAM
'''
def shortest_path_dfs(q,graph, start, end):

    try:
        return next(iterative_search(q, graph, start, end))
    except StopIteration:
        return None
'''
'----------------------------------------------------'
#declare local variable for shortest paths of bfs
gen_bfs = shortest_path_bfs(Queue.Queue(),my_graph, 1, 165)
'---------------------------------------------------------------------------------------------'
#local variable for our dataset
sourceFileName='C:\Users\Badr\PycharmProjects\Project_Metro_Paris\metro_paris.txt'
#dataset and dictionary
def dataset_with_dict():
    f = open(sourceFileName, 'r')
    my_l = []
    my_stations_with_dict = {}
    for unit in f.readlines()[1:]:
        if unit.strip() != '[Edges]':
            unit = unit.decode("ISO-8859-1").encode("utf-8") # encoding for utf8

            my_l.append(unit[5:])
        else:
            break
    for i in xrange(len(my_l)):
        my_stations_with_dict[i] = my_l[i]

    return my_stations_with_dict

#declare local variable for our function dataset_with_dict()
dict_my_stations = dataset_with_dict()
'--------------------------------------------------------------------------------------------'
#join shortest path of BFS (in our exemple) with dictionary
def short_path(l, dict_with_stations,b_flag=None):
    l_s_path = []
    for unit in l:
        l_s_path.append(("id:{0} - Station :<{1}>".format(unit, dict_with_stations[unit])))
    if b_flag:
        return '-'.join(l_s_path)
    else:
        return (l_s_path)

#declare local variable for our function short path using local varialbe of BFS algorithm with path between 1 and 165 and dictionary
short_path_printed = short_path(gen_bfs, dict_my_stations,"p")
'---------------------------------------------------------------------------------------------'

#shortest path of bfs
print 'Shortest path bfs'
print(shortest_path_bfs(Queue.Queue(),my_graph, 1, 165)),'\n'
'----------------------------------------------------'
#shortest path of dfs (need to fix memory error)
'''
print 'Shortest path dfs'
print(shortest_path_dfs(Stack(),my_graph, 1, 165))
'''
'----------------------------------------------------'
#shortest path of bfs with dictionary
print 'Shortest path bfs with dictionary'
print (short_path_printed)

'-------------------------------------------------------------------------------'
#Script for Dijkstra Algorithm
def dijkstra(graph,start,target):
    inf = 0
    for u in graph:  #sum(int(x) for x in a)
        for v ,w in graph[u]:
           inf = inf + w
    dist = dict([(u,inf) for u in graph])
    prev = dict([(u,None) for u in graph])
    q = graph.keys()
    dist[start] = 0
    #helper function
    def x(v):
        return dist[v]

    while q != []:
        u = min(q, key=x)
        q.remove(u)
        for v,w in graph[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    #way
    trav = []
    temp = target
    while temp != start:
        trav.append(prev[temp])
        temp = prev[temp]
    trav.reverse()
    trav.append(target)
    return trav,dist[target]  # if we want to use another example .join(map(str, trav)),dist[target]
'----------------------------------------------------'
#test our graph with pprint
#pprint(graph)
#pprint(my_graph_dijkstra)
'----------------------------------------------------'
#declare local variable for traverse of our way and distance between them
print
traverse = dijkstra(my_graph_dijkstra,1,165)[0]
dist = dijkstra(my_graph_dijkstra,1,165)[1]
'----------------------------------------------------'
print'---Shortest path Dijkstra Algorithm---'
#print traverse of our way
print traverse,'\n'
'----------------------------------------------------'
#declare local variable for only the way of our dijkstra algorithm , the way is "return trav" we need to use [0] to select only trav and not dist[target]
printed_drijkstra = dijkstra(my_graph_dijkstra,1,165)[0]
'----------------------------------------------------'
#declare local variable for our dijkstra algorithm with dictionary
short_path_printed_for_dijkstra = short_path(printed_drijkstra, dict_my_stations,"p")
'----------------------------------------------------'
#print dijkstra algorithm with dictionary
print (short_path_printed_for_dijkstra)
'----------------------------------------------------'
# call local variable distance between start and target of our shortest path Dijkstra algorithm
print 'Distance Between Start and Target for Shortest path Dijkstra Algorithm:' ,("%.0f" % dist),'\n'
'-----------------------------------------------------------------------------------------------------------------'
#create shortest path for dijkstra algorithm with binary heap (optimisation)
def shortestPath_binary_heap(graph, start, end):
    queue,seen = [(0, start, [])], set()
    while True:
        (cost, v, path) = heapq.heappop(queue)
        if v not in seen:
            path = path + [v]
            seen.add(v)
            if v == end:
                return cost, path
            newgraph=graph[v]
            for (next, c) in newgraph:# it seems that mydict.iteritems() is the best option here, as it gives you access to the values directly through tuple unpacking.
                heapq.heappush(queue, (cost + c, next, path))
'------------------------------------------------------------------------------------------------------------------'
print'---Shortest path Dijkstra Algorithm with Binary Heap---'
cost, path = shortestPath_binary_heap(my_graph_dijkstra, 1, 165)
'-----------------------------------------------------------------------------------------------------------------'
#print shortest path in binary heap method
print 'Shortest path binary heap :',path,'\n'
printed_drijkstra_binary_heap = shortestPath_binary_heap(my_graph_dijkstra, 1, 165)[1]
'-----------------------------------------------------------------------------------------------------------------'
#declare local variable for our dijkstra algorithm binary heap with dictionary
short_path_printed_for_dijkstra_binary_heap = short_path(printed_drijkstra_binary_heap, dict_my_stations,"p")
#print dijkstra algorithm binary heap with dictionary
print (short_path_printed_for_dijkstra_binary_heap)
'-----------------------------------------------------------------------------------------------------------------'
#print the cost between start and end of our graph in binary heap method
print 'Cost:',("%.0f" % cost),'\n'

'-----------------------------------------------------------------------------------------------------------------'
#measuring time
t1 = time()
print 'Measuring and comparing the running times of all algorithms : %f' %(t1-t0)
'-----------------------------------------------------------------------------------------------------------------'
#Don't enabled Fibonacci Heap code, still some issue to fix it

#------------------NOTE------------------------------------------------------------
'''
In short:

With a Fibonacci heap, good option but it's harder to code up.
With a binary heap,  is not a good option when dealing with large, dense graphs.
'''
'''
Bonus: Fibonacci Heap (not yet fixed)
#!/usr/bin/env python


from heapq import *
from random import randint
#  For HashMap
class Dijkstra(object):
    """ generated source for class Dijkstra """
    def shortestPaths(cls, graph, source):
        """ generated source for method shortestPaths """
        pq = FibonacciHeap()
        entries = HashMap()
        result = HashMap()
        for node in graph:
            entries.put(node, pq.enqueue(node, Double.POSITIVE_INFINITY))
        pq.decreaseKey(entries.get(source), 0.0)

        while not pq.isEmpty():
            result.put(curr.getValue(), curr.getPriority())
            for arc in graph.edgesFrom(curr.getValue()).entrySet():
                if result.containsKey(arc.getKey()):
                    continue
                if pathCost < dest.getPriority():
                    pq.decreaseKey(dest, pathCost)

        return result
'''

#Running time test Algorithm BFS with n = Step (100,300,500,700,900,1100) , Delete comments if you want tot test the running time with output of Linear Graphic
'''
'----------------------------------------------------'
ta= time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 100step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 100:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'
ta_end= time()
ta_time= (ta_end-ta)
print 'time 100step : ',ta_time
'----------------------------------------------------'
'----------------------------------------------------'
tb= time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 300step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 300:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'

tb_end= time()
tb_time= (tb_end-tb)
print 'time 300step : ',tb_time
'----------------------------------------------------'
'----------------------------------------------------'
tc = time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 500step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 500:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'

tc_end= time()
tc_time= (tc_end-tc)
print 'time 500step : ',tc_time
'----------------------------------------------------'
'----------------------------------------------------'
td = time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 700step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 700:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'

td_end= time()
td_time= (td_end-td)
print 'time 700step : ',td_time
'----------------------------------------------------'
'----------------------------------------------------'
te = time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 900step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 900:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'

te_end= time()
te_time= (te_end-te)
print 'time 900step : ',te_time
'----------------------------------------------------'
'----------------------------------------------------'
tf = time()
# bfs - using queue
gen = iterative_search(Queue.Queue(), my_graph, 1, 165)
print "---BFS 1100step---"'\n'
# get only 5 first paths
bfs_path_set = set()
while len(bfs_path_set) < 1100:
    bfs_path_set.add(next(gen))

print os.linesep.join(map(str, bfs_path_set)),'\n'

tf_end= time()
tf_time= (tf_end-tf)
print 'time 1100step : ',tf_time
'----------------------------------------------------'

print 'Running time of algorithm'
# make up some data
x = [100,300,500,700,900,1100]
y = [ta_time,tb_time,tc_time,td_time,te_time,tf_time]

# plot
plt.plot(x,y)
# beautify the x-labels
plt.gcf().autofmt_xdate()

plt.show()

print 'Running time done'
'''
