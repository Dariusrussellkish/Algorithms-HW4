# Algorithms-HW5

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

tests/test_graph.py::test_get_files PASSED          
tests/test_graph.py::test_graph_creation PASSED     
tests/test_graph.py::test_DFS PASSED                
tests/test_graph.py::test_SCC PASSED                
tests/test_graph.py::test_SCC_random PASSED
...                                                 
```

Testing of the module without `Networkx` installed is very limited (but you should have it from the conda environment).
There are only two files that test major cases for graph structure. 
*There is no support for SSSP algorithms without NetworkX.*

For testing, I use `Networkx` to generate Erdos-Renyi graphs of size 100 with edge probability of 0.5. Weights are then
added from a uniform distribution between [0, 1_000_000] or [-1_000_000, 1_000_000] as floats. This `NetworkX` graph is
then converted into my format. I then run the SCC algorithm to get the largest strongly connected component (guarantees
a path) and then two randomly selected vertices of that SCC are chosen and the SSSP algorithm run on it. I also run
reference implementations of the SSSP algorithms from `NetworkX`. I then compare both paths to ensure they are the same.
Because I use floating point numbers for weights, it's **highly** improbable there are two or more valid paths. 

