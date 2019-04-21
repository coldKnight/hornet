/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/ClosenessCentrality/clc.cuh"
#include "Static/ClosenessCentrality/exact_clc.cuh"
#include "Static/ClosenessCentrality/approximate_clc.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    // GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    // graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    Timer<DEVICE> TM;


    // graph.read(argv[1], SORT | PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    ClosenessCentrality clc(hornet_graph);

	vid_t root = graph.max_out_degree_id();
	if (argc==3)
	  root = atoi(argv[2]);
    // root = 226410;
    cout << "Root is " << root << endl;
    clc.reset();
    clc.setRoot(root);

    cudaProfilerStart();TM.start();
    clc.run();

    TM.stop();cudaProfilerStop();
    TM.print("ClosenessCentrality");
#if 0

    ExactCLC eclc(hornet_graph);

    eclc.reset();

    cudaProfilerStart();TM.start();
    // eclc.run();
    TM.stop();cudaProfilerStop();
    TM.print("Exact ClosenessCentrality");

    vid_t numRoots=1000;
    vid_t* roots = new vid_t[numRoots];
    ApproximateCLC::generateRandomRootsUniform(hornet_graph.nV(), numRoots, &roots, 1 );

    ApproximateCLC aclc(hornet_graph, roots,numRoots);
    aclc.reset();

    cudaProfilerStart();TM.start();
    // aclc.run();
    TM.stop();cudaProfilerStop();
    TM.print("Approximate ClosenessCentrality");


    delete[] roots;
#endif
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

