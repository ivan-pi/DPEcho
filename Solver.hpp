//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Solver_hpp_
#define _Solver_hpp_

#include "echo.hpp"
#include "utils/tb-types.hpp"

#include <sycl/sycl.hpp>

#if NRK==1
constexpr int RK_STAGES = 1;
const float crk1[] = {0.0};
const float crk2[] = {1.0};
#elif NRK==2
constexpr int RK_STAGES = 2;
const float crk1[] = {0.0, 0.5};
const float crk2[] = {1.0, 0.5};
#elif NRK==3
constexpr int RK_STAGES = 3;
const float crk1[] = {0.0, 0.75, 1.0/3.0};
const float crk2[] = {1.0, 0.25, 2.0/3.0};
#elif NRK==4
constexpr int RK_STAGES = 5;
// RK4(3)5[2R+]C, Kennedy/Carpenter/Lewis (low-storage, 2-register, 5 stages)
// `ark` has RK_STAGES-1 entries (a_i), `brk` has RK_STAGES entries (b_i).
const float ark[] = {
  970286171893.0f/4311952581923.0f,
  6584761158862.0f/12103376702013.0f,
  2251764453980.0f/15575788980749.0f,
  26877169314380.0f/34165994151039.0f
};
const float brk[] = {
  1153189308089.0f/22510343858157.0f,
  1772645290293.0f/4653164025191.0f,
 -1672844663538.0f/4480602732383.0f,
  2114624349019.0f/3568978502595.0f,
  5198255086312.0f/14908931495163.0f
};
#elif NRK==5
constexpr int RK_STAGES = 9;
// RK5(4)9[2R+]S, Kennedy/Carpenter/Lewis (low-storage, 2-register, 9 stages)
// `ark` has RK_STAGES-1 entries (a_i), `brk` has RK_STAGES entries (b_i).
const float ark[] = {
  1107026461565.0f/5417078080134.0f,
  38141181049399.0f/41724347789894.0f,
  493273079041.0f/11940823631197.0f,
  1851571280403.0f/6147804934346.0f,
  11782306865191.0f/62590030070788.0f,
  9452544825720.0f/13648368537481.0f,
  4435885630781.0f/26285702406235.0f,
  2357909744247.0f/11371140753790.0f
};
const float brk[] = {
  2274579626619.0f/23610510767302.0f,
  693987741272.0f/12394497460941.0f,
 -347131529483.0f/15096185902911.0f,
  1144057200723.0f/32081666971178.0f,
  1562491064753.0f/11797114684756.0f,
  13113619727965.0f/44346030145118.0f,
  393957816125.0f/7825732611452.0f,
  720647959663.0f/6565743875477.0f,
  3559252274877.0f/14424734981077.0f
};
#else
#error "NRK must be 1, 2, 3, 4, or 5"
#endif

// Only those called directly outside Solver.cpp needed.
SYCL_EXTERNAL field holibDer(int myId, field *var, int stride);
SYCL_EXTERNAL void  holibRec(int myId, field *var, int stride,  field *vL, field *vR); // Reconstructs at i +/- 1/2

#endif
