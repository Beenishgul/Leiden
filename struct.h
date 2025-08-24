#ifndef STRUCT_H
#define STRUCT_H

#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <unordered_map>
#include <map>
#include <chrono>


struct adjlist {
 std::unordered_map <int, std::vector <int> > out_gr;
 std::unordered_map<int,  std::vector <double> > out_wt;
 std::unordered_map <int, std::vector <int> > in_gr;
 std::unordered_map<int,  std::vector <double> > in_wt;
 int edges;
 int len;
};

struct Leiden_Partition {
  double* tot_out;
  double* tot_in;
  double* sum_in;
  double* in_deg;
  double* out_deg;
  int* node_comm;
  int* size;
  int* final_comm;
  double* home_comm;
  int* older_comm;
  double* sum_kin;
  double* self_loops;
  double weight;
  int imp[1];

  std::vector <int> neigh_commNb;
  std::vector <int> neigh_pos;
  std::vector <int> count;
  int *nbrs;
  int *pos;
};

struct graph {
  int* child_in;
  int* child_out;
  double* wts_out;
  double* wts_in;
  int* in_col;
  int* out_col;
  int nodes;
  int ed;
};

struct aggregate_adj{
std::vector<std::vector <std::pair <double, int> > > next_graph;
std::unordered_map <int, std::vector < std::pair <double, int> > > in_neighbours;
  
 std::vector <int>  out_gr;
 std::vector <double>  out_wt;
 std::vector <int>  in_gr;
 std::vector <double>  in_wt;

int edges;
int len;
int vertices;
};

 struct new_g_arrays{
 int* positions;
 int* fake_id;
 int* sorted_arr;
 int* real_comm;
 int* comms;
 int* indices;
};

struct arrays{ 
 std::vector <double> temp;
 std::vector <int> temp2;
 std::vector <int> row;
 std::vector <int> col;
 std::vector <double> outwt2;
 std::vector <int> outgr2;
};

graph graph_process (adjlist& adj, graph& g);
int Leiden_CPU(Leiden_Partition& p, graph& gr);

Leiden_Partition create_c_partition(graph& g, Leiden_Partition& p);

graph next_pahse(aggregate_adj& ad, graph& g);

int Leidencomplete(adjlist& g);

inline double selfloop(graph& g, Leiden_Partition& p, int v);

inline double outdegree(graph& g, Leiden_Partition& p, int v);

inline double indegree(graph& g, Leiden_Partition& p, int v);

double cal_quality(double in[], double tot_in[], double tot_out[],long int size, double edgs);

void free(graph& g);

void free_part(Leiden_Partition& p);

//int renumber_communities(Leiden_Partition& p, graph& g);

bool compareNodesByCommunity(int node1, int node2);
int c_renumber_communities(Leiden_Partition& p, graph& g);
//int find_neighbours(Leiden_Partition& p, graph& g, new_g_arrays& pre, const std::vector<Edge>& c_list, int V);






#endif