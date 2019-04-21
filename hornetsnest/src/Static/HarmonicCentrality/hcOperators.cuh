/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com
 * @author Vatsal Srivastava                                              <br>
 *   Georgia Institute of Technology                                        <br>       
 * @date April, 2019
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
 *
 * @file
 */

#include "Static/HarmonicCentrality/hc.cuh"

namespace hornets_nest {


// Used at the very beginning of every HC computation.
// Used only once.
struct InitHC {
    HostDeviceVar<hcData> hcd;


    OPERATOR(vid_t src) {
        hcd().hc[0] = 0.0;
    }
};



// Used at the very beginning of every hc computation.
// Once per root
struct InitOneTree {
    HostDeviceVar<hcData> hcd;

    // Used at the very beginning
    OPERATOR(vid_t src) {
        hcd().d[src] = INT32_MAX;
    }
};

struct hc_BFSTopDown {
    HostDeviceVar<hcData> hcd;

    OPERATOR(Vertex& src, Edge& edge){

        degree_t nextLevel = hcd().currLevel + 1;
		
        vid_t v = src.id(), w = edge.dst_id();        

        degree_t prev = atomicCAS(hcd().d + w, INT32_MAX, nextLevel);
        if (prev == INT32_MAX) {
            hcd().queue.insert(w);
        }
    }
};

// Used at the very beginning of every hc computation.
// Once per root
struct Incrementhc {
    HostDeviceVar<hcData> hcd;

    // Used at the very beginning
    OPERATOR(vid_t src) {
        if(src == hcd().root)
            hcd().hc[src]+=hcd().roothc;
    }
};

} // namespace hornets_nest
