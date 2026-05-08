//  Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//  Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Domain.hpp"
#include "Grid.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"

#include <array>

#if __has_include(<mdspan>)
#include <mdspan>
namespace mdspan_ns = std;
#elif __has_include(<experimental/mdspan>)
#include <experimental/mdspan>
namespace mdspan_ns = std::experimental;
#else
#error "DPEcho requires an mdspan implementation to build Domain::BCex."
#endif

using namespace sycl;

Domain::~Domain( ){
  free(bufL, qq); free(bufR, qq);
#ifdef MPICODE
#if MPICODE != SR_REPLACE
  free(sendBufL, qq); free(sendBufR, qq);
#endif
#if MPICODE == START
// These should be here, but give errors. Check again with future MPI releases.
//  MPI_Request_free(reqSendL);  MPI_Request_free(reqSendR);
//  MPI_Request_free(reqRecvL);  MPI_Request_free(reqRecvR);
#endif
#endif
  Log::cout(4) << TAG << "Domain removed and BCex buffers deallocated." << Log::endl;
}

#ifdef MPICODE
#define CART_DEFAULT 0
#else
#define CART_DEFAULT 1
#endif

Domain::Domain(sycl::queue q, size_t bufSizes[NDIM], Parameters &param) {
  int nprocs = Log::mpiSize(), myrank = Log::mpiRank(),  reorder = 0;
  size_t bufMax = std::max({bufSizes[0], bufSizes[1], bufSizes[2]});
  qq = q; // Use the queue to allocate the buffers
#if defined(MPICODE) && (MPICODE != SR_REPLACE)
  sendBufL= malloc_shared<field>(FLD_TOT*bufMax, qq); Log::Assert(sendBufL, "Cannot allocate recvBufL.");
  sendBufR= malloc_shared<field>(FLD_TOT*bufMax, qq); Log::Assert(sendBufR, "Cannot allocate recvBufR.");
#endif
  bufL= malloc_shared<field>(FLD_TOT*bufMax, qq);     Log::Assert(bufL, "Cannot allocate bufL.");
  bufR= malloc_shared<field>(FLD_TOT*bufMax, qq);     Log::Assert(bufR, "Cannot allocate bufR.");

  boxMin_[0] = param.getOr("xMin", -0.5); boxMin_[1] = param.getOr("yMin", -0.5); boxMin_[2] = param.getOr("zMin", -0.5);
  boxMax_[0] = param.getOr("xMax",  0.5); boxMax_[1] = param.getOr("yMax",  0.5); boxMax_[2] = param.getOr("zMax",  0.5);
  bcType_[0] = param.getOr("bcTypex", BCPER); bcType_[1] = param.getOr("bcTypey", BCPER); bcType_[2] = param.getOr("bcTypez", BCPER);
  cartDims_[0]=param.getOr("Rx", CART_DEFAULT); cartDims_[1] = param.getOr("Ry", CART_DEFAULT); cartDims_[2] = param.getOr("Rz", CART_DEFAULT);

  for(int iD=0; iD<NDIM; ++iD){ boxSize_[iD] = boxMax_[iD]-boxMin_[iD]; }

#ifndef MPICODE
  for(int i =0; i <NDIM; ++i ){ isEdgeLeft_[i] = isEdgeRight_[i] = cartDims_[i] = 1; cartCoords_[i] = 0;  }
#else
  // Create cartesian domain
  int cartPeriodic_[3]={1,1,1}; // MPI is broken and will never accept non-periodic, so we give it to it anyways.
  MPI_Dims_create(nprocs, 3, cartDims_);
  MPI_Cart_create(MPI_COMM_WORLD, 3, cartDims_, cartPeriodic_, reorder, &cartComm_);
  // Check that domain is consistent
  auto prod = cartDims_[0] * cartDims_[1] * cartDims_[2];
  if ( (prod) && (prod != nprocs) ){
    Log::cerr(0) << TAG << "Aborting. cartDims: " << prod << " don't match MPI ranks :" << nprocs << Log::endl;
    Log::Assert(false, "Terminating.");
  }
  MPI_Cart_coords(cartComm_, myrank, 3, cartCoords_);
  int neighCoords[3]; // One of those MPI messy interfaces. Bad code. At least it scales! - SC
  //-- Edges and neighbours
  for(unsigned short i=0; i<3; ++i){
    isEdgeLeft_ [i]=(              0 == cartCoords_[i])?1:0; // Left  edge.
    isEdgeRight_[i]=((cartDims_[i]-1)== cartCoords_[i])?1:0; // Right edge.
    for(unsigned short j=0; j<3; ++j){ neighCoords[j] = cartCoords_[j]; } // Reset
    neighCoords[i] = cartCoords_[i]-1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+i);
    neighCoords[i] = cartCoords_[i]+1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+i);
  }
#endif
  //-- Initialize phisical (local!) dimensions
  for(unsigned short i=0; i<NDIM; ++i){
    locSize_[i] = boxSize_[i]/cartDims_[i]; locMin_[i] = boxMin_[i]+locSize_[i]*cartCoords_[i]; locMax_[i] = locMin_[i]+locSize_[i];
  }
  Log::clog() << TAG << "Domain created!" << Log::endl;  cartInfo();

#if defined(MPICODE) && (MPICODE == START)
  for(unsigned short i=0; i<NDIM; ++i){
    MPI_Send_init(sendBufL,bufSizes[i]*FLD_TOT,MPI_FIELD,neighRankPrev_[i],10,cartComm_,&reqSendL[i]);
    MPI_Send_init(sendBufR,bufSizes[i]*FLD_TOT,MPI_FIELD,neighRankNext_[i],20,cartComm_,&reqSendR[i]);
    MPI_Recv_init(    bufL,bufSizes[i]*FLD_TOT,MPI_FIELD,neighRankNext_[i],10,cartComm_,&reqRecvL[i]);
    MPI_Recv_init(    bufR,bufSizes[i]*FLD_TOT,MPI_FIELD,neighRankPrev_[i],20,cartComm_,&reqRecvR[i]);
  }
#endif

}

void Domain::cartInfo() {
  Log::cout(1) << TAG << "3D Domain: "  << cartDims_[0] << " " << cartDims_[1] << " " << cartDims_[2] << Log::endl;
#ifdef MPICODE
  Log::clog(10) << TAG << "CartPrev: ("  << neighRankPrev_[0] << " " << neighRankPrev_[1] << " "<< neighRankPrev_[2] << ") "
                << "CartNext: (" << neighRankNext_[0] << " " << neighRankNext_[1] << " " << neighRankNext_[2] << ") " << Log::endl;
  int neighCoords_[3];
  for (int i = 0; i < 3; ++i){
    Log::clog(10) << TAG << "Prev / This / Next / Edge(l/r) [dir #" << i << "]" ;
    MPI_Cart_coords(cartComm_, neighRankPrev_[i], 3, neighCoords_);
    Log::clog(10) << "(" << neighCoords_[0] << " " << neighCoords_[1] << " " << neighCoords_[2] << ") ";
    Log::clog(10) << "(" << cartCoords_[0]  << " " << cartCoords_[1]  << " " << cartCoords_[2]  << ") ";
    MPI_Cart_coords(cartComm_, neighRankNext_[i], 3, neighCoords_);
    Log::clog(10) << "(" << neighCoords_[0] << " " << neighCoords_[1] << " " << neighCoords_[2] <<") ";
    Log::clog(10) << "(" << isEdgeLeft_[i]  << "/" << isEdgeRight_[i] <<") ";
    Log::clog(10) << Log::endl;
  }
#else
  Log::clog(5) << TAG << "CartCoords:    (" << cartCoords_[0] << " " << cartCoords_[1] << " " << cartCoords_[2] << ")" << Log::endl;
#endif
}

void Domain::boxInfo() {
  Log::cout(1) << TAG << "Box  size (" << boxSize_[0] << " " << boxSize_[1] << " " << boxSize_[2]
               << ") From ("   << boxMin_ [0] << " " << boxMin_ [1] << " " << boxMin_ [2]
               << ") To ("     << boxMax_ [0] << " " << boxMax_ [1] << " " << boxMax_ [2] << ")" << Log::endl;
}

void Domain::locInfo() {
  Log::clog(1) <<TAG << "Local size (" << locSize_[0] << " " << locSize_[1] << " " << locSize_[2]
               << ") From ("   << locMin_ [0] << " " << locMin_ [1] << " " << locMin_ [2]
               << ") To ("     << locMax_ [0] << " " << locMax_ [1] << " " << locMax_ [2] << ")" << Log::endl;
}

// Variables are passed, so any array can be used.
// ACHTUNG:
//  - It always assumes periodic, w or w/o MPI. At the end it will take care of other BCs.
void Domain::BCex(int myDir, Grid gr, field_array &v, int dType){ // gr is the usual local grid.

#ifdef MPICODE
  int myRank; MPI_Comm_rank(cartComm_, &myRank);
  MPI_Status  status;
#endif
  int i0 = (dType==BCEX_VU)?0:1; // Flux is Mx+1 so bcex needs shift

  // Each direction may have a different buffer size. We may also store them, whatever.
  int   nBuf[]=      {gr.nh[0], gr.nh[1], gr.nh[2]}; nBuf[myDir] = gr.h[myDir];
  int   nOff[]=      {       0,        0,        0}; nOff[myDir] = gr.h[myDir];
  range rBuf  = range( nBuf[0],  nBuf[1],  nBuf[2]); auto   sBuf = rBuf.size(); auto rPlane=rBuf; // rPlane is for BCOF3
  nd_range<3> ndr = getMatchingNdRange(rBuf, range<3>(4,4,4));
  Log::clog(8) << TAG << " Filling buffers ..." << Log::endl;

  field *bL = this->bufL, *bR = this->bufR;
#ifdef MPICODE
#if MPICODE != SR_REPLACE
  bL = this->sendBufL; bR = this->sendBufR;
#endif
#if   MPICODE == ISEND
  MPI_Irecv(bufL,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],10,cartComm_,&reqRecvL[myDir]);
  MPI_Irecv(bufR,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],20,cartComm_,&reqRecvR[myDir]);
#elif MPICODE == START
  MPI_Start(&reqRecvL[myDir]);
  MPI_Start(&reqRecvR[myDir]);
#endif
#endif
  id<3> nOffRead = range<3>(nOff[0], nOff[1], nOff[2]); nOffRead[myDir] +=i0;
  range<3> const fullGrR(gr.nh[0], gr.nh[1], gr.nh[2]);
  using mdspan_extent_1d = mdspan_ns::extents<size_t, mdspan_ns::dynamic_extent>;
  using mdspan_extent_3d = mdspan_ns::extents<size_t, mdspan_ns::dynamic_extent, mdspan_ns::dynamic_extent, mdspan_ns::dynamic_extent>;
  using field_view = mdspan_ns::mdspan<field, mdspan_extent_1d>;
  using buffer_view = mdspan_ns::mdspan<field, mdspan_ns::extents<size_t, FLD_TOT, mdspan_ns::dynamic_extent>>;
  using layout_mapping = mdspan_ns::layout_right::mapping<mdspan_extent_3d>;
  const size_t fullFieldExtent = static_cast<size_t>(gr.nht);
  const std::array<field_view, FLD_TOT> fieldViews{
    field_view(v[0], fullFieldExtent), field_view(v[1], fullFieldExtent), field_view(v[2], fullFieldExtent), field_view(v[3], fullFieldExtent),
    field_view(v[4], fullFieldExtent), field_view(v[5], fullFieldExtent), field_view(v[6], fullFieldExtent), field_view(v[7], fullFieldExtent)
  };
  const layout_mapping bufMap(mdspan_extent_3d(static_cast<size_t>(nBuf[0]), static_cast<size_t>(nBuf[1]), static_cast<size_t>(nBuf[2])));
  const layout_mapping fullGridMap(mdspan_extent_3d(static_cast<size_t>(gr.nh[0]), static_cast<size_t>(gr.nh[1]), static_cast<size_t>(gr.nh[2])));
  const auto linearBufId = [=](id<3> const idx) -> size_t {
    return bufMap(idx[0], idx[1], idx[2]);
  };
  const auto reverseLinearBufId = [=](id<3> const idx) -> size_t {
    return bufMap(rBuf[0] - 1 - idx[0], rBuf[1] - 1 - idx[1], rBuf[2] - 1 - idx[2]);
  };
  const auto linearGridId = [=](id<3> const idx, id<3> const offset) -> size_t {
    return fullGridMap(idx[0] + offset[0], idx[1] + offset[1], idx[2] + offset[2]);
  };
  const auto reverseLinearGridId = [=](id<3> const idx, id<3> const offset) -> size_t {
    return fullGridMap(fullGrR[0] - 1 - (idx[0] + offset[0]), fullGrR[1] - 1 - (idx[1] + offset[1]), fullGrR[2] - 1 - (idx[2] + offset[2]));
  };
  const auto linearGridIdNoOffset = [=](id<3> const idx) -> size_t {
    return fullGridMap(idx[0], idx[1], idx[2]);
  };
  const buffer_view leftSendView(bL, static_cast<size_t>(sBuf));
  const buffer_view rightSendView(bR, static_cast<size_t>(sBuf));
  qq.parallel_for( ndr, [=](nd_item<3> it){
    id<3> id = it.get_global_id();
    if (isOutOfBounds(id, rBuf)) return;
    size_t iBufL = linearBufId(id), iBufR = reverseLinearBufId(id);
    size_t iVL   = linearGridId(id, nOffRead), iVR   = reverseLinearGridId(id, nOffRead);
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      leftSendView(iVar, iBufL) = fieldViews[iVar](iVL);
#if defined(MPICODE) && ( (MPICODE == ISEND) || (MPICODE == START) )
    }
  }).wait_and_throw();
#if   MPICODE == ISEND
  MPI_Isend(sendBufL,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],10,cartComm_,&reqSendL[myDir]);
#elif MPICODE == START
  MPI_Start(&reqSendL[myDir]);
#endif
  qq.parallel_for( rBuf, [=](item<3> it){
    id<3> id = it.get_id();
    size_t iBufL = linearBufId(id), iBufR = reverseLinearBufId(id);
    size_t iVL   = linearGridId(id, nOffRead), iVR   = reverseLinearGridId(id, nOffRead);
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
#endif
      rightSendView(iVar, iBufR) = fieldViews[iVar](iVR);
    }
  }).wait_and_throw();

#ifdef MPICODE //- Middle communication
#if   MPICODE == ISEND
  MPI_Isend(sendBufR,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],20,cartComm_,&reqSendR[myDir]);
  MPI_Wait (&reqRecvL[myDir],&status);
#elif MPICODE == START
  MPI_Start(&reqSendR[myDir]);
  MPI_Wait (&reqRecvL[myDir],&status);
#elif MPICODE == SENDRECV
 Log::clog(8) << TAG << "L MPI_Sendrrecv ... " << Log::endl;
  MPI_Sendrecv(
    sendBufL,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],10,
        bufL,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],10,
    cartComm_,&status);
  Log::clog(8) << TAG << "R MPI_Sendrrecv ... " << Log::endl;
  MPI_Sendrecv(
    sendBufR,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],20,
        bufR,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],20,
    cartComm_,&status);
#elif MPICODE == SR_REPLACE
  Log::clog(8) << TAG << "L+R MPI_Sendrrecv_replace ... " << Log::endl;
  MPI_Sendrecv_replace(bufL,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],myRank,neighRankNext_[myDir],MPI_ANY_TAG,cartComm_,&status);
  MPI_Sendrecv_replace(bufR,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],myRank,neighRankPrev_[myDir],MPI_ANY_TAG,cartComm_,&status);
#endif
#endif
  Log::clog(8) << TAG << " Recopying from buffers..." << Log::endl;
  nOff[myDir] = 0; // To write, we start from 0
  id<3> nOffW = range<3>(nOff[0], nOff[1], nOff[2]);
  const buffer_view leftRecvView(this->bufL, static_cast<size_t>(sBuf));
  const buffer_view rightRecvView(this->bufR, static_cast<size_t>(sBuf));
  qq.parallel_for(ndr,[=,bL=this->bufL,bR=this->bufR](nd_item<3> it){  // v -> WHindex
    id<3> id = it.get_global_id();
    if (isOutOfBounds(id, rBuf)) return;
    size_t iVL   = linearGridId(id, nOffW), iVR   = reverseLinearGridId(id, nOffW);  // For the regular BCEX
    size_t iBufL = linearBufId(id), iBufR = reverseLinearBufId(id);  // The same, if we start from the end
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      fieldViews[iVar](iVR) = leftRecvView(iVar, iBufR);  // ...besides the flipped assignments
#if defined(MPICODE) && ( (MPICODE == ISEND) || (MPICODE == START) )
    }
  }); // NO SYCL wait here!
  MPI_Wait (&reqRecvR[myDir],&status);
  qq.parallel_for(rBuf,[=,bL=this->bufL,bR=this->bufR](item<3> it){  // v -> WHindex
    id<3> id = it.get_id();
    size_t iVL   = linearGridId(id, nOffW), iVR   = reverseLinearGridId(id, nOffW);  // For the regular BCEX
    size_t iBufL = linearBufId(id), iBufR = reverseLinearBufId(id);  // The same, if we start from the end
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
#endif
      fieldViews[iVar](iVL) = rightRecvView(iVar, iBufL);  // ACHTUNG: Must reverse both L<-->R and the indexes in them!
    }
  }).wait_and_throw();
  switch(bcType_[myDir]){ //-- PROCESSING BC TYPEs
    case BCOF0: //- Outflow w. 0th order interp
      Log::clog(10) << TAG << " Processing Outflow BCs of order 0..." << Log::endl;
      if(isEdgeLeft_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> readId, writeId = readId = it.get_id()  ;  readId[myDir]+= gr.h[myDir] - it.get_id(myDir);
          size_t iVL = linearGridIdNoOffset(writeId), iOut = linearGridIdNoOffset(readId);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){
            fieldViews[iVar](iVL) = fieldViews[iVar](iOut);
          };
        }).wait_and_throw();
      }
      if(isEdgeRight_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> gridOffset = id(0,0,0)                             ; gridOffset[myDir] =   gr.n[myDir] + gr.h[myDir];
          id<3> readId, writeId = readId = it.get_id() + gridOffset;     readId[myDir] = readId[myDir] - it.get_id(myDir) - 1;
          size_t iVR = linearGridIdNoOffset(writeId), iOut = linearGridIdNoOffset(readId);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){
            fieldViews[iVar](iVR) = fieldViews[iVar](iOut);
          };
        }).wait_and_throw();
      }
      break;
      // ACHTUNG:: MOST LIKELY INCORRECT!!!!
    case BCOF3: //- Outflow w. 3rd order interp
      Log::clog(10) << TAG << " Processing Outflow BCs of order 3..." << Log::endl;
      if(isEdgeLeft_[myDir]){
        for(int iVar=0; iVar<FLD_TOT; ++iVar){
          qq.parallel_for(rPlane,[=](item<3> it){  // v -> WHindex
            id<3> myId = it.get_id();
            int iVL = static_cast<int>(linearGridIdNoOffset(myId));
            int step = stride(myId, myDir, gr.nh);
            for(int iLay=0; iLay<gr.h[myDir]; ++iLay){
               fieldViews[iVar](static_cast<size_t>(iVL)) = -1*fieldViews[iVar](static_cast<size_t>(iVL+step))
                                                           -3*fieldViews[iVar](static_cast<size_t>(iVL+2*step))
                                                           +  fieldViews[iVar](static_cast<size_t>(iVL+3*step));
               iVL+=-step;
              }
            });
        }; qq.wait_and_throw();
      }
      if(isEdgeRight_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> gridOffset = id(0,0,0);  gridOffset[myDir] = gr.n[myDir] + gr.h[myDir];
          id<3> myId = it.get_id() + gridOffset;
          int iVL = static_cast<int>(linearGridIdNoOffset(myId));
          int step = stride(myId, myDir, gr.nh);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){
            for(int iLay=0; iLay<gr.h[myDir]; ++iLay){
               fieldViews[iVar](static_cast<size_t>(iVL)) = -1*fieldViews[iVar](static_cast<size_t>(iVL-step))
                                                           -3*fieldViews[iVar](static_cast<size_t>(iVL-2*step))
                                                           +  fieldViews[iVar](static_cast<size_t>(iVL-3*step));
               iVL+= step;
            }
          }
        }).wait_and_throw();
      }
      break;
    case BCPER:  break; //- Periodic; do nothing
    default   : Log::cerr(2) << TAG << "Unknown BC TYPE " << bcType_[myDir] << " along direction " << myDir << ". Proceeding as periodic." << Log::endl;
  }// End switch

#if defined(MPICODE) && ( (MPICODE == ISEND) || (MPICODE == START) )
  MPI_Wait(&reqSendL[myDir],&status);
  MPI_Wait(&reqSendR[myDir],&status);
#endif
  Log::clog(8) << TAG << " BCex complete." << Log::endl;
  return;
}
