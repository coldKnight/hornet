/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/HarmonicCentrality/hc.cuh"
#include "Static/HarmonicCentrality/exact_hc.cuh"
#include "Static/HarmonicCentrality/approximate_hc.cuh"
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

	HarmonicCentrality hc(hornet_graph);

	vid_t root = graph.max_out_degree_id();
	if (argc==3)
	  root = atoi(argv[2]);
    // root = 226410;
    cout << "Root is " << root << endl;
    hc.reset();
    hc.setRoot(root);

    cudaProfilerStart();TM.start();
    hc.run();

    TM.stop();cudaProfilerStop();
    TM.print("HarmonicCentrality");
#if 0

    ExactHC ehc(hornet_graph);

    ehc.reset();

    cudaProfilerStart();TM.start();
    // ehc.run();
    TM.stop();cudaProfilerStop();
    TM.print("Exact HarmonicCentrality");

    vid_t numRoots=1000;
    vid_t* roots = new vid_t[numRoots];
    ApproximateHC::generateRandomRootsUniform(hornet_graph.nV(), numRoots, &roots, 1 );

    ApproximateHC ahc(hornet_graph, roots,numRoots);
    ahc.reset();

    cudaProfilerStart();TM.start();
    // ahc.run();
    TM.stop();cudaProfilerStop();
    TM.print("Approximate HarmonicCentrality");


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

