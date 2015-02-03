//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_ENCODER_H_
#define CRFPP_ENCODER_H_

#include "common.h"
#ifdef USE_MPI
#include "mpi_comm.h"
#endif  // USE_MPI

namespace CRFPP {
class Encoder {
 public:
  enum { CRF_L2, CRF_L1, MIRA };
  bool learn(const char *, const char *,
             const char *,
             bool, size_t, size_t,
             double, double,
             unsigned short,
             unsigned short, int
#ifdef USE_MPI
             , const std::vector<std::string> &
#endif  // USE_MPI
             );

  bool convert(const char *text_file,
               const char *binary_file);

  const char* what() { return what_.str(); }

#ifdef USE_MPI
  void setMpiComm(MpiComm *comm) { comm_ = comm; }
#endif  // USE_MPI

 private:
  whatlog what_;
#ifdef USE_MPI
  uint8_t data_part_;
  MpiComm *comm_;
#endif  // USE_MPI
};
}
#endif
