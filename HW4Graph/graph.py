from collections import defaultdict
from copy import deepcopy


class Node:
    def __init__(self, id_):
        self.id = id_
        self.start = None
        self.finish = None

    def __copy__(self):  # shallow copy if you want to reset the graph
        return Node(self.id)

    def __deepcopy__(self, memo):  # retain start/finish, for SCC calculation
        ret = Node(self.id)
        ret.start = self.start
        ret.finish = self.finish
        return ret


class Graph:
    def __init__(self, adj_list):
        self.graph = defaultdict(list)  # could be done with ints in guaranteed O(1), assume dict is O(1) lookup
        id_to_vertex = dict()
        for line in adj_list:  # convert adj list to dict structure
            if not line:
                continue
            try:  # try/except may not be needed anymore
                node_1 = id_to_vertex[line[0]]
            except KeyError:
                node_1 = Node(line[0])
                id_to_vertex[line[0]] = node_1
                _ = self.graph[node_1]  # ensure node with no neighbors is in the graph
            if len(line) > 1:  # ensure we have neighbors to iterate over
                for n2 in line[1:]:
                    try:
                        node_2 = id_to_vertex[n2]
                    except KeyError:
                        node_2 = Node(n2)
                        id_to_vertex[n2] = node_2
                    self.graph[node_1].append(node_2)
                    _ = self.graph[node_2]  # ensure ill-defined adj lists are valid

    @staticmethod
    def from_file(filepath):
        with open(filepath, 'r') as fh:
            return Graph.from_strlike(fh)

    @staticmethod
    def from_strlike(str):
        adj_list = []
        for line in str:
            nodes = line.strip().split()
            adj_list.append(nodes)
        return Graph(adj_list)
    
    @property
    def edges(self):
        if hasattr(self, 'edges_'):
            return self.edges_
        edges = []
        for node in self:
            for neighbor in self[node]:
                edges.append((node, neighbor))
        setattr(self, 'edges_', edges)
        return self.edges_
    
    @property
    def nodes(self):
        if hasattr(self, 'nodes_'):
            return self.nodes_
        setattr(self, 'nodes_', list(self.graph.keys()))
        return self.nodes_

    def __getitem__(self, item):
        return self.graph[item]

    def __iter__(self):
        for node in self.graph.keys():
            yield node

    def __len__(self):
        return len(self.graph)

    @property
    def T(self):  # Inversion with lazy evaluation. O(V + E)
        if hasattr(self, 'rev'):
            return self.rev
        reversed_dict = defaultdict(list)
        setattr(self, 'rev', reversed_dict)
        for node1 in self:
            for node2 in self[node1]:
                reversed_dict[node2].append(node1)
            _ = reversed_dict[node1]
        ret = deepcopy(self)
        ret.graph = reversed_dict
        return ret


def new_counter(start_val=1):  # increment via closure
    val = start_val

    def postinc():
        nonlocal val
        val += 1
        return val - 1
    return postinc


def DFS(G: Graph, sort_by_finish=False, return_finished=False, order=None):
    if order is None:
        order = []
    visited = defaultdict(bool)  # all nodes are unvisited
    finished_order = []
    postinc = new_counter()
    nodes = G.nodes
    if sort_by_finish:  # for SCC
        nodes = reversed(order)  # O(V)
    for node in nodes:  # O(V + E)
        if visited[node]:  # don't top level recurse on visited nodes
            continue
        rch = {node}
        visited[node] = True
        reachables = __DFS_body__(G, node, visited, postinc, finished_order,
                                  sort_by_finish=sort_by_finish, order=order)
        rch |= reachables  # size of component, sums to the same as O(V + E)
        if return_finished:
            yield rch, finished_order
        else:
            yield rch


def __DFS_body__(G: Graph, node: Node, visited, postinc, finished_order, sort_by_finish=False, order=None):
    rch = {node}
    node.start = postinc()
    for neighbor in G[node]:
        if visited[neighbor]:
            continue
        visited[neighbor] = True
        reachables = __DFS_body__(G, neighbor, visited, postinc, finished_order,
                                  sort_by_finish=sort_by_finish, order=order)
        rch |= reachables
    node.finish = postinc()
    finished_order.append(node)  # O(V) overall since this is only hit once per node
    return rch


def strongly_connected_components(G: Graph):
    if len(G) == 0:
        return

    finished = None
    for _, f in DFS(G, return_finished=True):  # O(V + E)
        finished = f
    G = G.T  # O(V + E)
    for component in DFS(G, sort_by_finish=True, order=finished):  # O(V + E)
        yield component
