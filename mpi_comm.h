#ifndef MPI_COMM_H_
#define MPI_COMM_H_

#include <stdint.h>
#include <vector>

#include "common.h"
#include "mpi.h"

class MpiComm {
public:
    MpiComm(bool debug) : m_buffer(NULL), m_flag(0), m_debug(debug) {
        m_buffer = new char [kMaxBufferSize];
        memset(m_buffer, 0, kMaxBufferSize);
    }

    ~MpiComm() {
        if (NULL != m_buffer) {
            delete [] m_buffer;
            m_buffer = NULL;
        }
    }

    bool IsDebug() { return m_debug; }

    bool SendWeightToWorker(const double *w, const WorkerInfo &worker_info, int worker_rank, bool orthant);
    bool SendGradientObjToMaster(const double *g, uint32_t num, double o, int worker_tag);

    bool RecvWeightFromMaster(bool orthant, uint32_t target_num, double *w);
    bool RecvGradientObjFromWorker(const std::vector<WorkerInfo> &workers_info, int worker_rank,
            double *g, double *o, std::vector<uint8_t> &rank_2_part);

    void SetFlag() { m_flag = 1; }
    int GetFlag() { return m_flag; }
    bool Bcast() {
        return (MPI_SUCCESS == MPI_Bcast(&m_flag, 1, MPI_INT, kMasterRank, MPI_COMM_WORLD));
    }

private:
    MPI_Status m_status;
    char *m_buffer;
    int m_flag;  // whether to stop
    bool m_debug;  // whether to print detail training info

    static const uint32_t kMsgTag = 1;
    static const uint32_t kMasterRank = 0;
    static const uint32_t kMaxBufferSize = 10000000 * 8;
};

#endif  // MPI_COMM_H_

