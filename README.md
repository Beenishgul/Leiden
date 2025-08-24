# Leiden
Parallel implementation Leidem Algorithm on GPUs

# **Introduction**
The Leiden algorithm is a popular method for community detection in networks, improving upon the Louvain algorithm by guaranteeing well-connected communities. While most existing implementations focus on undirected graphs and run on CPUs, this project introduces a GPU-accelerated implementation of the Leiden algorithm for directed graphs.
#
This implementation leverages CUDA to parallelize the key phases of the Leiden algorithm—local movement, refinement, and partition aggregation—making it significantly faster and scalable for large directed networks.


