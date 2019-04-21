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

#include "Static/HarmonicCentrality/hc.cuh"
#include "Static/HarmonicCentrality/exact_hc.cuh"

using namespace std;
namespace hornets_nest {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in
// the streaming case.

ExactHC::ExactHC(HornetGraph& hornet) :
                                       HCentrality(hornet)
{
    start_v = 0;
    stop_v  = hornet.nV();

    reset();
}

ExactHC::~ExactHC() {
    release();
}

void ExactHC::reset() {
    HCentrality::reset();
}

void ExactHC::release(){
}


void ExactHC::run() {

    for(vid_t r=start_v; r<stop_v; r++){
        if((r%200)==0)
            cout << r << ", " << flush;
        HCentrality::setRoot(r);
        HCentrality::run();
    }
    cout << endl;
}


bool ExactHC::validate() {
    return true;
}

} // namespace hornets_nest
