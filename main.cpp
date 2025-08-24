#include "struct.h"
#include <thread> 
#include "leiden.h"


using namespace std;


int main(int argc, char* argv[])
{

  if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

string filename = argv[1];
string mode = argv[2];
std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
ifstream file(filename);
auto start_time = std::chrono::high_resolution_clock::now();
    if (!file.is_open()) {
        cout << "Failed to open the file: " << filename << endl;
        return 1;
    }

adjlist adj;
    adj.edges = 0;

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int v1, v2;
        float w;
while (iss >> v1 >> v2 >> w) {
            adj.out_gr[v1].push_back(v2);
            adj.out_wt[v1].push_back(w);
            adj.in_gr[v2].push_back(v1);
            adj.in_wt[v2].push_back(w);
            adj.edges++;
        }
}
    file.close();


adj.len=adj.out_gr.size();


 auto end_time = std::chrono::high_resolution_clock::now();
 auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
 std::cout << "Time taken for reading first graph is " << duration.count() << " minutes!" << std::endl;

 Leiden_Partition part;
 graph g;
 g.nodes= adj.len;
 graph_process(adj, g);
 g.ed=adj.edges;
 create_c_partition(g,part);
 int arr_size= adj.edges;

if (mode == "cpu") {
    Leiden_CPU(part, g);
} else if (mode == "gpu") {
    Leiden_GPU(part, g, arr_size);

} else {
    cout << "Invalid mode. Use 'cpu' or 'gpu'." << endl;
    return 1;
}

 free(g);
 free_part(part);
                                                                                          
    return 0;
}



