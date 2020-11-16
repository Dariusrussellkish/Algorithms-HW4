# Algorithms-HW4

### To Install (ensuring you have conda)
```bash
conda env create --file environment.yaml
conda env activate algorithms
pip install -e .
```

### To Test

```bash
pytest -vvv -s
```

You should see the following output plus additional verbose printing
```text
collected 5 items                                                                                                                                                          

tests/test_graph.py::test_get_files PASSED                                                                                                                           [ 20%]
tests/test_graph.py::test_graph_creation PASSED                                                                                                                      [ 40%]
tests/test_graph.py::test_DFS PASSED                                                                                                                                 [ 60%]
tests/test_graph.py::test_SCC PASSED                                                                                                                                 [ 80%]
tests/test_graph.py::test_SCC_random PASSED                                                                                                                          [100%]
```

Testing of the module without `Networkx` installed is very limited (but you should have it from the conda environment).
There are only two files that test major cases for graph structure. 

With `Networkx`, I take advantage of its random graph generation and its SCC implementation to use
as a reference, and `assert` that my calculated SCC is the same as theirs for 1000 random binomial graphs of (100, 0.2).

The implementation is able to run in $O(V + E)$ time because we can get an ordering of the nodes from DFS without
needing to sort. Every other operation is a DFS or graph reversal, both of which are $O(V + E)$