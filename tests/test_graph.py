import glob
import os
from pathlib import Path

import pytest
import networkx

from HW4Graph.graph import Graph, DFS
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
    nx = pytest.importorskip('networkx', reason="Networkx is not installed")

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
