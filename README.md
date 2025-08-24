# Leiden
Parallel implementation Leidem Algorithm on GPUs

# **Introduction**
The Leiden algorithm is a popular method for community detection in networks, improving upon the Louvain algorithm by guaranteeing well-connected communities. While most existing implementations focus on undirected graphs and run on CPUs, this project introduces a GPU-accelerated implementation of the Leiden algorithm for directed graphs.
#
This implementation leverages CUDA to parallelize the key phases of the Leiden algorithm—local movement, refinement, and partition aggregation—making it significantly faster and scalable for large directed networks.

## Prerequisites
- **C++ compiler**: g++ (version >= 9 recommended)
- **CUDA Toolkit**: version >= 12.0
- **CMake** (optional, if using CMake for building)
- **Make** utility
- **Linux environment** (tested on Ubuntu 20.04)
- **Optional libraries**:
  - `libm` (math library, usually comes with gcc)

# **Input Format**
#Graph in the following format:
# source target weight
# 0 1 1.0
# 1 2 2.5
# 2 0 0.7


