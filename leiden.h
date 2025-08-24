// leiden.h

#ifndef LEIDEN_H
#define LEIDEN_H
#include <stdio.h>
#include "struct.h"

struct gpu_partition {
  double* tot_out;
  double* tot_in;
  double* sum_in;
  double* in_deg;
  double* out_deg;
  int* node_comm;
  int* final_comm;
  int* size;
  int* home_comm;
  int* older_comm;
  double* sum_kin;
  double* self_loops;
  double weight;
  size_t node_comm_size;
  int *nbrs;
  int *pos;
   int *neigh_commNb;
    int *neigh_pos;
};
  struct gpu_graph {
  int* child_in;
  int* child_out;
  double* wts_out;
  double* wts_in;
  int* in_col;
  int* out_col;
  int nodes;
};


Leiden_Partition create_partition(graph& g, Leiden_Partition& p);
int Leiden_GPU(Leiden_Partition& p, graph& g, int E);
double find_quality(Leiden_Partition& p, graph& g);
double ToOwnCommunity(int node, int community, double bestGain, int old_comm, Leiden_Partition& d_p, graph& d_g);
double computGain(int node, int community, Leiden_Partition& d_p, graph& d_g);
double find_to_own(Leiden_Partition& d_p, graph& d_g, double dncomm, int i, int community, int comm);

#endif // LEIDEN_H