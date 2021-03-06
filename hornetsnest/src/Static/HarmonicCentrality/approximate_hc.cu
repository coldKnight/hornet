/**
 * @brief
 * @author Oded Green                                                       <br>
 *   NVIDIA Corporation                                                     <br>       
 *   ogreen@nvidia.com                                                      <br>       
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
#include <Core/GPUHornet/Hornet.cuh>

#include "Static/HarmonicCentrality/hc.cuh"
#include "Static/HarmonicCentrality/approximate_hc.cuh"

#include <stdio.h>
#include <stdlib.h>

using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

ApproximateHC::ApproximateHC(HornetGraph& hornet, 
    vid_t* h_rootIDs_, vid_t numRoots_):
                                       HCentrality(hornet)
{
    numRoots = numRoots_;
    h_rootIDs = new vid_t[numRoots];
    memcpy(h_rootIDs,h_rootIDs_, sizeof(vid_t)*numRoots);

    reset();
}

ApproximateHC::~ApproximateHC() {
    release();
}

void ApproximateHC::reset() {
    HCentrality::reset();
}

void ApproximateHC::release(){
    delete [] h_rootIDs;
}

void ApproximateHC::run() {

    for(vid_t r=0; r<numRoots; r++){
        if((r%200)==0)
            cout << r << ", " << flush;
        
        HCentrality::setRoot(h_rootIDs[r]);
        HCentrality::run();
    }
    cout << endl;
}


bool ApproximateHC::validate() {
    return true;
}

    
void ApproximateHC::generateRandomRootsUniform(vid_t nV, 
    vid_t numRoots, vid_t** returnRoots, int RandSeed){

    bool* selected = new bool[nV];

    vid_t* tempRoots = new vid_t[numRoots];
    memset(selected,false,sizeof(bool)*nV);

    time_t t;

    if (RandSeed==-1)
        srand((unsigned) time(&t));

    int v=0;
    while(v<numRoots){
        vid_t randV = rand()%nV;

        if(selected[randV]==true)
            continue;
        selected[randV]=true;
        tempRoots[v++]=randV;

    }

    delete[] selected;
    *returnRoots = tempRoots;
}


} // namespace hornets_nest
