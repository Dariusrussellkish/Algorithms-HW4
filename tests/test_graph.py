import glob
import os
import random
from collections import defaultdict
from pathlib import Path

import pytest
import networkx

from HW4Graph.graph import Graph, DFS, dijkstra, bellman_ford, NegativeEdgeCycleError, find_path_or_error
from HW4Graph.graph import strongly_connected_components as scc


def get_test_files():
    path = os.path.dirname(os.path.abspath(__file__)) / Path('data')
    for file in glob.glob(str(path / Path("*.txt"))):
        yield file


def test_get_files():
    print()
    print("### BEGIN TEST get_files() ###")
    for file in get_test_files():
        print(file)
    print("### END TEST get_files() ###")


def test_graph_creation():
    print()
    print("### BEGIN TEST test_graph_creation() ###")
    for file in get_test_files():
        print(file)
        G = Graph.from_file(file)
        for node in G:
            neighbors = list(G[node])
            print(f"{node.id} : {[n.id for n in neighbors]}")

        print("Reversing")
        G = G.T
        for node in G:
            neighbors = list(G[node])
            print(f"{node.id} : {[n.id for n in neighbors]}")
    print("### END TEST test_graph_creation() ###")


def test_DFS():
    print()
    print("### BEGIN TEST test_DFS() ###")
    for file in get_test_files():
        print(file)
        G = Graph.from_file(file)
        f = None
        for rch, finished in DFS(G, return_finished=True):
            f = finished
            print([n.id for n in rch])
            # print([(n.id, v) for (n, v) in visited.items()])
        print([n.id for n in f])
        for node in G:
            print(f"{node.id}: start: {node.start:4d} finish: {node.finish:4d}")
    print("### END TEST test_DFS() ###")


def test_SCC():
    print()
    print("### BEGIN TEST test_SCC() ###")
    for file in get_test_files():
        print(file)
        G = Graph.from_file(file)
        for component in scc(G):
            print([n.id for n in component])
    print("### END TEST test_SCC() ###")


def test_SCC_random():
    pytest.importorskip('networkx', reason="Networkx is not installed")

    print()
    print("### BEGIN TEST test_SCC_random() ###")
    print("Expect test to take a minute to elapse but nothing to print")
    for _ in range(1000):
        # Create a random digraph
        G = networkx.binomial_graph(100, 0.2, directed=True)
        adj_list = [line for line in networkx.generate_adjlist(G)]

        # use networkx reference for SCCs
        sccs = list(networkx.strongly_connected_components(G))
        ref_sccs = set()
        for scc_ in sccs:
            ref_sccs.add(frozenset([v for v in scc_]))
        R = Graph.from_strlike(adj_list)
        our_sccs = set()
        for component in scc(R):
            our_sccs.add(frozenset(int(n.id) for n in component))

        # lots of work to make them easily comparable
        assert our_sccs == ref_sccs
    print("### END TEST test_SCC_random() ###")


def generate_random_weighted_graph(n, p, weight_min=0.):
    G = networkx.gnp_random_graph(n, p, directed=True)
    g = defaultdict(list)
    for (u, v, w) in G.edges(data=True):
        weight = random.uniform(weight_min, 1_000_000.0)
        w['weight'] = weight
        g[u].append((weight, v))

    adj_list = []
    for node, nbrs in g.items():
        adj_list.append([node] + [nbr for nbr in nbrs])

    return G, Graph(adj_list, weighted=True)


def test_dijkstra():
    pytest.importorskip('networkx', reason="Networkx is not installed")
    print()
    print("### BEGIN TEST test_dijkstra() ###")
    print("Expect test to take a minute")
    for _ in range(100):
        nG, G = generate_random_weighted_graph(100, 0.2)
        sccs = [list(c) for c in scc(G)]
        sccs.sort(key=len, reverse=True)
        c = sccs[0]
        s, t = random.choices(c, k=2)
        preds, dists = networkx.dijkstra_predecessor_and_distance(nG, int(s.id))
        nx_path = list(find_path_or_error(preds, int(s.id), int(t.id)))
        our_path = [n.id for n in dijkstra(G, s, t)]
        print(f"NetworkX Path: {nx_path}")
        print(f"Our Path     : {our_path}")
        assert nx_path == our_path

    print("### END TEST test_dijkstra() ###")


def test_bellman_ford():
    pytest.importorskip('networkx', reason="Networkx is not installed")
    print()
    print("### BEGIN TEST test_bellman_ford() ###")
    print("Expect test to take a minute")
    for _ in range(100):
        nG, G = generate_random_weighted_graph(100, 0.5, weight_min=-10000.0)
        sccs = [list(c) for c in scc(G)]
        sccs.sort(key=len, reverse=True)
        c = sccs[0]
        s, t = random.choices(c, k=2)
        try:
            p, dist = networkx.bellman_ford_predecessor_and_distance(nG, int(s.id), target=int(t.id))
            print(f"NetworkX Path: {list(find_path_or_error(p, int(s.id), int(t.id)))}")
            print(f"Our Path     : {[n.id for n in bellman_ford(G, s, t)]}")
        except networkx.NetworkXUnbounded:
            try:
                print((s.id, t.id), [n.id for n in bellman_ford(G, s, t)])
                assert False, "NetworkX predicts negative edge cycle, we don't"
            except NegativeEdgeCycleError:
                print("NetworkX predicts negative edge cycle, so do we")
                assert True, "NetworkX predicts negative edge cycle, so do we"
    print("### END TEST test_bellman_ford() ###")
