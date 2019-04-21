/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 * @author Vatsal Srivastava                                              <br>
 *   Georgia Institute of Technology                                        <br>       
 * @date April, 2019
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


#include "Static/HarmonicCentrality/hc.cuh"

#include "hcOperators.cuh"

using length_t = int;
using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

HCentrality::HCentrality(HornetGraph& hornet) :
                                       StaticAlgorithm(hornet),
                                       load_balancing(hornet)
{
    hd_hcData().currLevel=0;
    hd_hcData().levelSum=0.0;
    hd_hcData().roothc=0.0;
    
    cout << "hornet.nV   " << hornet.nV() << endl;

    gpu::allocate(hd_hcData().d, hornet.nV());
	gpu::allocate(hd_hcData().hc, hornet.nV());
    hd_hcData().queue.initialize(hornet);

    reset();
}

HCentrality::~HCentrality() {
    release();
}

void HCentrality::reset() {
    hd_hcData().currLevel=0;
    hd_hcData().levelSum=0.0;
    hd_hcData().roothc=0.0;

    forAllnumV(hornet, InitHC { hd_hcData });
    forAllnumV(hornet, InitOneTree { hd_hcData });
    hd_hcData.sync();
}

void HCentrality::release(){
    gpu::free(hd_hcData().d);
    gpu::free(hd_hcData().hc);
}

void HCentrality::setRoot(vid_t root_){
    hd_hcData().root=root_;
}

void HCentrality::run() {


    // Initialization
    hd_hcData().currLevel=0;
    hd_hcData().levelSum=0.0;
    hd_hcData().roothc=0.0;
	
    forAllnumV(hornet, InitOneTree { hd_hcData });
    vid_t root = hd_hcData().root;

    hd_hcData().queue.insert(root);                   // insert source in the frontier
    gpu::memsetZero(hd_hcData().d + root);


    while (hd_hcData().queue.size() > 0) {
		
        forAllEdges(hornet, hd_hcData().queue, hc_BFSTopDown { hd_hcData }, load_balancing);
		if (hd_hcData().currLevel > 0){
			// HC = (1/distance) * (number of nodes)
			hd_hcData().levelSum += (hc_t) ((hc_t) (hd_hcData().queue.size())) / ((hc_t) (hd_hcData().currLevel));
		}
        //cout << hd_hcData().currLevel << " x " << hd_hcData().queue.size() << endl;
        
        hd_hcData().currLevel++;

        hd_hcData().queue.swap();
    }
	
    forAllnumV(hornet, Incrementhc { hd_hcData });
    hd_hcData().roothc = (hc_t) (hd_hcData().levelSum)/((hc_t) hornet.nV() - 1.0) ;
	//cout << "hc " << hd_hcData().roothc << endl;
    //cout << "levelSum " << hd_hcData().levelSum << endl;
}

bool HCentrality::validate() {
    return true;
}

} // namespace hornets_nest
