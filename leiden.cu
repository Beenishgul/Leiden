
#include <cuda_runtime.h>
#include "leiden.h"
#include <iostream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iomanip>

using namespace std;

# define cuCALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
   }
}


__device__ double find_to_own(Leiden_Partition& d_p, graph& d_g, double dncomm,
                              int i, int community, int comm, int chk)
{
    // Iterate over outgoing edges of node i
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        int target = d_g.child_out[neighbour];
        if (i != target && d_p.node_comm[target] == comm) {
            dncomm += d_g.wts_out[neighbour];
        }
    }

    // Iterate over incoming edges of node i
    for (int neighbor = d_g.in_col[i]; neighbor < d_g.in_col[i + 1]; neighbor++) {
        int target = d_g.child_in[neighbor];
        if (i != target && d_p.node_comm[target] == comm) {
            dncomm += d_g.wts_in[neighbor];
        }
    }

    return dncomm;
}


__device__ double to_community(Leiden_Partition& d_p, graph& d_g, int i, int best_comm, double dncomm)
{
    // Iterate over outgoing edges
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        if (d_g.child_out[neighbour] < i) {
            if (i != d_g.child_out[neighbour] && d_p.node_comm[d_g.child_out[neighbour]] == best_comm) {
                dncomm += d_g.wts_out[neighbour];
            }
        }
        else if (d_g.child_out[neighbour] > i && d_p.older_comm[d_g.child_out[neighbour]] == best_comm) {
            dncomm += d_g.wts_out[neighbour];
        }
    }

    // Iterate over incoming edges
    for (int neighbour = d_g.in_col[i]; neighbour < d_g.in_col[i + 1]; neighbour++) {
        if (d_g.child_in[neighbour] < i) {
            if (i != d_g.child_in[neighbour] && d_p.node_comm[d_g.child_in[neighbour]] == best_comm) {
                dncomm += d_g.wts_in[neighbour];
            }
        }
        else if (d_g.child_in[neighbour] > i && d_p.older_comm[d_g.child_in[neighbour]] == best_comm) {
            dncomm += d_g.wts_in[neighbour];
        }
    }

    return dncomm;
}


__device__ double removal(Leiden_Partition& d_p, graph& d_g, double dnc, int i, int comm)
{
    // Iterate over outgoing edges
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        if (i != d_g.child_out[neighbour]) {
            if (d_g.child_out[neighbour] < i && d_p.node_comm[d_g.child_out[neighbour]] == comm) {
                dnc += d_g.wts_out[neighbour];
            }
            else if (d_g.child_out[neighbour] > i && d_p.older_comm[d_g.child_out[neighbour]] == comm) {
                dnc += d_g.wts_out[neighbour];
            }
        }
    }

    // Iterate over incoming edges
    for (int neighbour = d_g.in_col[i]; neighbour < d_g.in_col[i + 1]; neighbour++) {
        if (i != d_g.child_in[neighbour]) {
            if (d_g.child_in[neighbour] < i && d_p.node_comm[d_g.child_in[neighbour]] == comm) {
                dnc += d_g.wts_in[neighbour];
            }
            else if (d_g.child_in[neighbour] > i && d_p.older_comm[d_g.child_in[neighbour]] == comm) {
                dnc += d_g.wts_in[neighbour];
            }
        }
    }

    return dnc;
}


__device__ int update_weights(Leiden_Partition& d_p, graph& d_g, int i, double dnc)
{
    if (d_p.node_comm[i] != d_p.older_comm[i]) {
        // Update sum_in for old and new communities using atomic operations
        atomicAdd(&d_p.sum_in[d_p.older_comm[i]], -(dnc + d_p.self_loops[i]));  // Remove contribution from old community
        atomicAdd(&d_p.sum_in[d_p.node_comm[i]], d_p.home_comm[i] + d_p.self_loops[i]); // Add contribution to new community
    }

    // Update total degrees for old and new communities
    atomicAdd(&d_p.tot_in[d_p.older_comm[i]],  -d_p.in_deg[i]);
    atomicAdd(&d_p.tot_out[d_p.older_comm[i]], -d_p.out_deg[i]);
    atomicAdd(&d_p.tot_in[d_p.node_comm[i]],    d_p.in_deg[i]);
    atomicAdd(&d_p.tot_out[d_p.node_comm[i]],   d_p.out_deg[i]);

    return 0;
}


__global__ void update_partition(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < d_g.nodes)
    {
        int best_comm = d_p.final_comm[i];
        d_p.node_comm[i] = best_comm;

        double dncomm = 0.0;

        // Compute weight to the new community
        dncomm = to_community(d_p, d_g, i, best_comm, dncomm);

        if (best_comm != d_p.older_comm[i])
        {
            atomicAdd(&d_p.home_comm[i], dncomm);
        }

        double dnc = 0.0;

        // Compute weight removal from old community
        dnc = removal(d_p, d_g, dnc, i, d_p.older_comm[i]);

        // Update community weights
        update_weights(d_p, d_g, i, dnc);
    }
}

__global__ void find_community(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double quality = 0.0;
    double newGain = 0.0;
    int best_comm, old_comm, mvs;
    double imp = 0.0;
    double prev_quality = 0.0;
    double q_prev_it = 0;
    double edg_wt = d_p.weight;

    if (i < d_g.nodes)
    {
        double dnc;
        double dnc2 = 0.0;
        double bestGain = 0.0;

        old_comm = d_p.node_comm[i];
        d_p.older_comm[i] = d_p.node_comm[i];
        d_p.home_comm[i] = 0;

        double toc_in = 0;
        double toc_out = 0;
        int comm = 0;
        double VertexToCommunity = 0.0;

        // Iterate over neighboring communities
        for (int community = d_p.pos[i]; community < d_p.pos[i + 1]; community++) 
        { 
            comm = d_p.older_comm[d_p.nbrs[community]];
            double dncomm = 0.0;

            dncomm = find_to_own(d_p, d_g, dncomm, i, community, comm, d_p.nbrs[community]);

            if (d_p.older_comm[i] == comm)
            {
                toc_in  = d_p.tot_in[comm]  - d_p.in_deg[i];
                toc_out = d_p.tot_out[comm] - d_p.out_deg[i];
            }
            else
            {
                toc_in  = d_p.tot_in[comm];
                toc_out = d_p.tot_out[comm];
            }

            newGain = (dncomm + d_p.self_loops[i]) / d_p.weight
                      - ((toc_in * d_p.out_deg[i] + toc_out * d_p.in_deg[i]) / (d_p.weight * d_p.weight));

            if (newGain > bestGain)                         
            { 
                bestGain = newGain;
                best_comm = comm;
                VertexToCommunity = dncomm;
            } 

            d_p.final_comm[i] = best_comm;
        }

        // Prevent swapping back and forth in some conditions
        if (d_p.final_comm[i] < d_p.older_comm[i] &&
            d_p.final_comm[d_p.final_comm[i]] == d_p.older_comm[i])
        {
            d_p.final_comm[i] = d_p.older_comm[i];
        } 

        if (d_p.size[d_p.older_comm[i]] > d_p.size[d_p.final_comm[i]] &&
            d_p.size[d_p.final_comm[i]] < d_p.size[d_p.older_comm[i]])
        {
            d_p.final_comm[i] = d_p.older_comm[i];
        } 

        // Update sizes atomically if community has changed
        if (d_p.final_comm[i] != d_p.older_comm[i])
        {
            atomicSub(&d_p.size[d_p.older_comm[i]], 1);
            atomicAdd(&d_p.size[d_p.final_comm[i]], 1);
        }
    }
}

 
double find_quality(Leiden_Partition& p, graph& g)
{
    double q = 0.0;

    for (int i = 0; i < g.nodes; i++)
    {
        if (p.tot_in[i] > 0 || p.tot_out[i] > 0) 
        {
            q += p.sum_in[i] - (p.tot_in[i] * p.tot_out[i] / p.weight);
        }
    }

    q = q / p.weight;
    return q;
}

int renumber_communities(Leiden_Partition& p, graph& g)
{
    unordered_map<int, vector<int>> comms;            

    // Group nodes by their community
    for (int comm = 0; comm < g.nodes; comm++)
    {
        comms[p.node_comm[comm]].push_back(comm);
    }

    // Re-number communities and remove duplicates
    vector<int> renumber_c;     
    int community_range = 0;

    for (auto it = comms.begin(); it != comms.end(); ++it)
    {
        renumber_c.push_back(it->first);
        community_range++;
    }

    sort(renumber_c.begin(), renumber_c.end());

    unordered_map<int, int> dummy_indexes;                   
    for (int it = 0; it < renumber_c.size(); it++)
    {
        dummy_indexes[renumber_c[it]] = it;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    aggregate_adj adj;

    // Build aggregated graph
    for (int comm = 0; comm < community_range; comm++)                                                             
    {
        int community = renumber_c[comm];
        const vector<int>& community_nodes = comms[community];

        unordered_map<int, double> temp;
        vector<pair<double, int>> temp_edges;

        for (int node = 0; node < community_nodes.size(); node++)                                                                        
        {
            for (int neighbor = g.out_col[community_nodes[node]]; 
                 neighbor < g.out_col[community_nodes[node] + 1]; neighbor++)                                             
            {
                int n_neig = g.child_out[neighbor]; 
                double w_neig = g.wts_out[neighbor];
                int neig_comm = p.node_comm[n_neig];
                temp[dummy_indexes[neig_comm]] += w_neig;    
            }
        } 

        for (const auto& entry : temp) 
        {
            temp_edges.push_back(make_pair(entry.second, entry.first));
        }      

        auto myCompare = [](const pair<double, int>& a, const pair<double, int>& b) {
            return a.second < b.second;
        };

        sort(temp_edges.begin(), temp_edges.end(), myCompare);                           
        adj.next_graph.push_back(temp_edges);  
    } 

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Aggregate step on Device completed in " << duration.count() << " minutes!" << std::endl; 
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;

    int arcs = 0;

    auto start_time2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < adj.next_graph.size(); i++)
    {
        for (int j = 0; j < adj.next_graph[i].size(); j++)
        {
            adj.in_neighbours[adj.next_graph[i][j].second].push_back({adj.next_graph[i][j].first, i});
        }

        arcs += adj.next_graph[i].size();
    }

    g.nodes = adj.next_graph.size();
    adj.len = adj.next_graph.size();
    adj.edges = arcs;
    g.ed = arcs;

    g.out_col = new int[(adj.len + 1)];
    g.in_col  = new int[(adj.len + 1)];
    g.out_col[0] = 0; 

    int valu = 0;
    for (int idx = 0; idx < adj.next_graph.size(); idx++) 
    {
        valu += adj.next_graph[idx].size();
        g.out_col[idx + 1] = valu;
    }

    g.in_col[0] = 0; 
    int s = 0;
    for (int idx = 0; idx < adj.in_neighbours.size(); idx++) 
    {
        s += adj.in_neighbours[idx].size();
        g.in_col[idx + 1] = s;
    }

    g.child_out = new int[adj.edges];
    g.child_in  = new int[adj.edges];
    g.wts_out   = new double[adj.edges];
    g.wts_in    = new double[adj.edges];

    int weight_index = 0;
    int node_index = 0;
    for (int i = 0; i < adj.next_graph.size(); i++)
    { 
        for (int j = 0; j < adj.next_graph[i].size(); j++)
        {
            g.child_out[node_index++] = adj.next_graph[i][j].second;
            g.wts_out[weight_index++] = adj.next_graph[i][j].first;
        }
    }

    int weight_index1 = 0;
    int node_index1 = 0;
    for (int i = 0; i < adj.in_neighbours.size(); i++)
    { 
        const vector<pair<double, int>> neighs = adj.in_neighbours[i];
        for (int j = 0; j < neighs.size(); j++)
        {
            g.child_in[node_index1++] = neighs[j].second;
            g.wts_in[weight_index1++] = neighs[j].first;
        }
    }

    create_partition(g, p);
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::minutes>(end_time2 - start_time2);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Preprocessing on Host completed in " << duration2.count() << " minutes!" << std::endl; 
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;
    
    Leiden_GPU(p, g, g.ed);

    return 0;   
}


Leiden_Partition create_partition(graph& g, Leiden_Partition& p)
{  
    // Allocate arrays
    p.in_deg      = new double[g.nodes];
    p.out_deg     = new double[g.nodes];
    p.tot_in      = new double[g.nodes];
    p.tot_out     = new double[g.nodes];
    p.sum_in      = new double[g.nodes];
    p.size        = new int[g.nodes];
    p.sum_kin     = new double[g.nodes];
    p.self_loops  = new double[g.nodes];
    p.node_comm   = new int[g.nodes];
    p.home_comm   = new double[g.nodes];
    p.older_comm  = new int[g.nodes];
    p.final_comm  = new int[g.nodes];
    p.nbrs        = new int[g.nodes + g.ed];
    p.pos         = new int[g.nodes + 1];

    // Initialize arrays
    for (int i = 0; i < g.nodes; i++)
    { 
        p.node_comm[i] = i;
        p.final_comm[i] = i;
        p.size[i] = 1;
        p.home_comm[i] = 0;
        p.older_comm[i] = i;
        p.in_deg[i] = 0;
        p.out_deg[i] = 0;
        p.tot_in[i] = 0;
        p.tot_out[i] = 0;
        p.sum_kin[i] = 0;
        p.self_loops[i] = 0;
        p.sum_in[i] = 0;
        p.weight = 0;
    }

    for (int i = 0; i < (g.nodes + g.ed); i++)
    { 
        p.nbrs[i] = 0;
    }

    for (int i = 0; i < (g.nodes + 1); i++)
    { 
        p.pos[i] = 0;
    }

 
    for (int i = 0; i < g.nodes; i++)
    { 
        p.in_deg[i] = indegree(g, p, i);
        p.out_deg[i] = outdegree(g, p, i);
        p.weight += p.out_deg[i];
        p.tot_in[i] = p.in_deg[i];
        p.tot_out[i] = p.out_deg[i];
        p.self_loops[i] = selfloop(g, p, i);
        p.sum_in[i] = p.self_loops[i];
    }


    p.count.clear();
    p.neigh_commNb.clear();
    p.neigh_pos.clear();

    int tot_inc = 0;
    p.count.push_back(0);

    for (int j = 0; j < g.nodes; j++)
    {
        int inc = 0;
        p.neigh_commNb.push_back(j);

        for (int i = g.out_col[j]; i < g.out_col[j + 1]; i++)
        {
            if (j != p.node_comm[g.child_out[i]])
            {
                p.neigh_commNb.push_back(p.node_comm[g.child_out[i]]);
                inc++;
            }
        }

        tot_inc += inc;
        p.count.push_back(tot_inc);
    }

    // Copy neighbor information
    for (int i = 0; i < p.neigh_commNb.size(); i++)
    {
        p.nbrs[i] = p.neigh_commNb[i];
    }

    p.neigh_pos.push_back(0); 
    for (int j = 1; j < g.nodes + 1; j++)
    {
        int b = p.count[j] - p.count[j - 1];
        int index = p.neigh_pos[j - 1];
        p.neigh_pos.push_back(index + b + 1);
    }

    for (int i = 0; i < p.neigh_pos.size(); i++)
    {
        p.pos[i] = p.neigh_pos[i];
    }

    return p;
}

inline double selfloop(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.out_col[v]; j < g.out_col[v + 1]; j++)
    {  
        if (v == g.child_out[j])
        {
            p.self_loops[v] += g.wts_out[j];
        }
    }  
    return p.self_loops[v];
}

inline double indegree(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.in_col[v]; j < g.in_col[v + 1]; j++)
    {
        p.in_deg[v] += g.wts_in[j];
    }  
    return p.in_deg[v];
}

inline double outdegree(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.out_col[v]; j < g.out_col[v + 1]; j++)
    {
        p.out_deg[v] += g.wts_out[j];
    }  
    return p.out_deg[v];
}

int Leiden_GPU(Leiden_Partition& p, graph& g, int E)
{
    Leiden_Partition d_p; 
    graph d_g;

    d_p.weight = p.weight;
    d_g.nodes = g.nodes;
    int V = g.nodes;

    auto start_time = std::chrono::high_resolution_clock::now();
    double quality = 0.0;
    double imp = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_p.node_comm, V * sizeof(int));
    cudaMalloc((void**)&d_p.size, V * sizeof(int));
    cudaMalloc((void**)&d_p.home_comm, V * sizeof(double));
    cudaMalloc((void**)&d_p.older_comm, V * sizeof(int));
    cudaMalloc((void**)&d_p.final_comm, V * sizeof(int)); 
    cudaMalloc((void**)&d_p.in_deg, V * sizeof(double));
    cudaMalloc((void**)&d_p.out_deg, V * sizeof(double));
    cudaMalloc((void**)&d_p.tot_in, V * sizeof(double));
    cudaMalloc((void**)&d_p.tot_out, V * sizeof(double));
    cudaMalloc((void**)&d_p.sum_in, V * sizeof(double));
    cudaMalloc((void**)&d_p.self_loops, V * sizeof(double));
    cudaMalloc((void**)&d_g.child_in, E * sizeof(int));
    cudaMalloc((void**)&d_g.child_out, E * sizeof(int));
    cudaMalloc((void**)&d_g.wts_in, E * sizeof(double));
    cudaMalloc((void**)&d_g.wts_out, E * sizeof(double));
    cudaMalloc((void**)&d_g.in_col, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_g.out_col, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_p.nbrs, (V + E) * sizeof(int));
    cudaMalloc((void**)&d_p.pos, (V + 1) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_p.node_comm, p.node_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.size, p.size, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.older_comm, p.older_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.home_comm, p.home_comm, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.final_comm, p.final_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.in_deg, p.in_deg, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.out_deg, p.out_deg, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.tot_in, p.tot_in, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.tot_out, p.tot_out, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.sum_in, p.sum_in, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.self_loops, p.self_loops, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.child_in, g.child_in, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.child_out, g.child_out, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.wts_in, g.wts_in, E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.wts_out, g.wts_out, E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.in_col, g.in_col, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.out_col, g.out_col, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.nbrs, p.nbrs, (V + E) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.pos, p.pos, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int tpb = 512;
    int nbl = (g.nodes + tpb - 1) / tpb;

    int moves = 0;
    double prev_quality = 0.0;
    double q_prev_it = 0;
    quality = find_quality(p, g);
    q_prev_it = quality;
    printf("previous quality: %f\n", q_prev_it);

    // Main Leiden iteration loop
    do
    {
        moves = 0;
        prev_quality = quality;

        // Community kernels
        find_community <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();

        update_partition <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();

        // Copy data back from GPU
        cudaMemcpy(p.node_comm, d_p.node_comm, V * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(p.sum_in, d_p.sum_in, V * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p.tot_in, d_p.tot_in, V * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p.tot_out, d_p.tot_out, V * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(p.older_comm, d_p.older_comm, V * sizeof(int), cudaMemcpyDeviceToHost);

        for (int comm = 0; comm < g.nodes; comm++)
        {
            if (p.node_comm[comm] != p.older_comm[comm])
            {
                moves++;
            }
        }
        quality = find_quality(p, g);
        imp = quality - prev_quality;
        printf("new quality: %.6f  imp = %.6f\n", quality, imp);

    } while (moves > 0 && imp > 0.005);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    std::cout << "Leiden step completed in " << duration.count() << " minutes!" << std::endl; 
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;

    if (quality > q_prev_it) 
    {                        
        renumber_communities(p, g);
    }
    else
    {
        cout << "Leiden_GPU done and dusted :)" << endl;
    }

    return 0;
}
