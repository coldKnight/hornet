/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 *   @author Muhammad Osama Sakhi                                           <br>
 *   Georgia Institute of Technology                                        <br>       
 * @date July, 2018
 *
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */


#include "Static/ClosenessCentrality/clc.cuh"

#include "clcOperators.cuh"

using length_t = int;
using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

CLCentrality::CLCentrality(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{
    hd_CLCData().currLevel=0;
    hd_CLCData().levelSum=0;
    hd_CLCData().rootClc=0.0;
    
    cout << "hornet.nV   " << hornet.nV() << endl;

    gpu::allocate(hd_CLCData().d, hornet.nV());
	gpu::allocate(hd_CLCData().clc, hornet.nV());
    hd_CLCData().queue.initialize(hornet);

    reset();
}

CLCentrality::~CLCentrality() {
    release();
}

void CLCentrality::reset() {
    hd_CLCData().currLevel=0;
    hd_CLCData().levelSum=0;
    hd_CLCData().rootClc=0.0;

    forAllnumV(hornet, InitCLC { hd_CLCData });
    forAllnumV(hornet, InitOneTree { hd_CLCData });
    hd_CLCData.sync();
}

void CLCentrality::release(){
    gpu::free(hd_CLCData().d);
    gpu::free(hd_CLCData().clc);
}

void CLCentrality::setRoot(vid_t root_){
    hd_CLCData().root=root_;
}

void CLCentrality::run() {


    // Initialization
    hd_CLCData().currLevel=0;
    hd_CLCData().levelSum=0;
    hd_CLCData().rootClc=0.0;
	
    forAllnumV(hornet, InitOneTree { hd_CLCData });
    vid_t root = hd_CLCData().root;

    hd_CLCData().queue.insert(root);                   // insert source in the frontier
    gpu::memsetZero(hd_CLCData().d + root);


    while (hd_CLCData().queue.size() > 0) {
		
        forAllEdges(hornet, hd_CLCData().queue, CLC_BFSTopDown { hd_CLCData }, load_balancing);
        hd_CLCData().levelSum += (degree_t) (hd_CLCData().currLevel) * (degree_t) (hd_CLCData().queue.size());
        //cout << hd_CLCData().currLevel << " x " << hd_CLCData().queue.size() << endl;
        
        hd_CLCData().currLevel++;

        hd_CLCData().queue.swap();
    }
	
    forAllnumV(hornet, IncrementCLC { hd_CLCData });
    hd_CLCData().rootClc = (clc_t) ((clc_t) hornet.nV() - 1.0)/ (clc_t) hd_CLCData().levelSum;
	//cout << "clc " << hd_CLCData().rootClc << endl;
    //cout << "levelSum " << hd_CLCData().levelSum << endl;
}

bool CLCentrality::validate() {
    return true;
}

} // namespace hornets_nest
