#include "mpi_comm.h"

#include <iomanip>

#include "common.h"

#define SPY_NUM 50

// format:
//   L2: num(4B) + {w(8B)}+
//   L1: num(4B) + {id(4B) + w(8B)}+
bool MpiComm::SendWeightToWorker(const double *w, const WorkerInfo &worker_info,
        int worker_rank, bool orthant) {
    std::ostringstream oss;
    oss << "[master][SendWeightToWorker] worker_rank:" << worker_rank << ", orthant:" << orthant
        << ", function_num:" << worker_info.feature_function_num << "\n";
    size_t offset = 4;
    uint32_t nonzero_weight_num = 0;
    const std::map<uint32_t, std::pair<uint32_t, uint8_t> > &ids_map =
        worker_info.ids_map;
    oss << "[master][SendWeightToWorker] worker_rank:" << worker_rank << ", weights:";
    int step = 0;
    for (std::map<uint32_t, std::pair<uint32_t, uint8_t> >::const_iterator it = ids_map.begin();
            it != ids_map.end(); ++it) {
        uint32_t l_id = it->first;
        uint32_t g_id = it->second.first;
        uint8_t function_num = it->second.second;
        if (orthant) {
            for (uint8_t i = 0; i < function_num; ++i) {
                uint32_t local_id = l_id + i;
                uint32_t global_id = g_id + i;
                if (!CRFPP::zero(w[global_id])) {
                    ++nonzero_weight_num;
                    memcpy(m_buffer + offset, reinterpret_cast<char *>(&local_id), sizeof(local_id));
                    offset += sizeof(local_id);
                    memcpy(m_buffer + offset, reinterpret_cast<char *>(
                                const_cast<double *>(&w[global_id])), sizeof(w[0]));
                    offset += sizeof(w[0]);
                    if (step < SPY_NUM) {
                        oss << "<" << local_id << "|" << global_id << "|"
                            << std::fixed << std::setprecision(9) << w[global_id] << ">, ";
                        ++step;
                    }
                }
            }
        } else {
            for (uint8_t i = 0; i < function_num; ++i) {
                uint32_t global_id = g_id + i;
                memcpy(m_buffer + offset, reinterpret_cast<char *>(
                            const_cast<double *>(&w[global_id])), sizeof(w[0]));
                offset += sizeof(w[0]);
                if (step < SPY_NUM) {
                    oss << "<" << l_id << "|" << global_id << "|"
                        << std::fixed << std::setprecision(9) << w[global_id] << ">, ";
                    ++step;
                }
            }
        }
    }
    oss << "\n";
    if (!orthant) nonzero_weight_num = worker_info.feature_function_num;
    oss << "[master][SendWeightToWorker] worker_rank:" << worker_rank << ", offset:" << offset
        << ", nonzero_weight_num:" << nonzero_weight_num << "\n";
    memcpy(m_buffer, reinterpret_cast<char *>(&nonzero_weight_num), sizeof(nonzero_weight_num));
    oss << "[master][SendWeightToWorker] worker_rank:" << worker_rank << ", begin to send\n";
    int ret = MPI_Send(m_buffer, offset, MPI_CHAR, worker_rank, kMsgTag, MPI_COMM_WORLD);
    oss << "[master][SendWeightToWorker] worker_rank:" << worker_rank << ", finish\n";
    std::cout << oss.str();
    return (MPI_SUCCESS == ret);
}

// format: num(4B) + obj(8B) + {g(8B)}{num}
// worker_tag: data part id loaded by this worker
bool MpiComm::SendGradientObjToMaster(const double *g, uint32_t num, double o, int worker_tag) {
    std::ostringstream oss;
    oss << "[worker][SendGradientObjToMaster] worker_tag:" << worker_tag << "\n";
    size_t offset = 0;
    memcpy(m_buffer + offset, reinterpret_cast<char *>(&num), sizeof(num));
    offset += sizeof(num);
    memcpy(m_buffer + offset, reinterpret_cast<char *>(&o), sizeof(o));
    offset += sizeof(o);
    oss << "[worker][SendGradientObjToMaster] worker_tag:" << worker_tag << ", num:" << num
        << ", obj:" << std::fixed << std::setprecision(9) << o << ", gradients:";
    for (uint32_t i = 0; i < num; ++i) {
        memcpy(m_buffer + offset, reinterpret_cast<char *>(
                    const_cast<double *>(&g[i])), sizeof(g[0]));
        offset += sizeof(g[0]);
        if (i < SPY_NUM) oss << std::fixed << std::setprecision(9) << g[i] << ", ";
    }
    oss << "\n";
    oss << "[worker][SendGradientObjToMaster] worker_tag:" << worker_tag << ", offset:" << offset << "\n";
    int ret = MPI_Send(m_buffer, offset, MPI_CHAR, kMasterRank, worker_tag, MPI_COMM_WORLD);
    oss << "[worker][SendGradientObjToMaster] worker_tag:" << worker_tag << ", finish\n";
    std::cout << oss.str();
    return (MPI_SUCCESS == ret);
}

bool MpiComm::RecvWeightFromMaster(bool orthant, uint32_t target_num, double *w) {
    std::ostringstream oss;
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", orthant:" << orthant << "\n";
    int char_num = 0;
    MPI_Recv(m_buffer, kMaxBufferSize, MPI_CHAR, kMasterRank, kMsgTag, MPI_COMM_WORLD, &m_status);
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", recv return\n";
    MPI_Get_count(&m_status, MPI_CHAR, &char_num);
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", char_num:" << char_num << "\n";

    // check integrity of received data
    uint32_t recv_num = 0;
    memcpy(&recv_num, m_buffer, sizeof(recv_num));
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", recv_num:" << recv_num << "\n";
    if (!orthant) {
        if (recv_num != target_num || recv_num * 8 != (uint32_t)(char_num - 4)) return false;
    } else {
        if (recv_num > target_num || recv_num * (4 + 8) != (uint32_t)(char_num - 4)) return false;
    }
    // read received weights
    size_t offset = 4;
    uint32_t fid = 0;
    double fval = 0.0;
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", weights:";
    if (orthant) {
        for (uint32_t i = 0; i < recv_num; ++i) {
            memcpy(&fid, m_buffer + offset, sizeof(fid));
            offset += sizeof(fid);
            if (fid >= target_num) return false;
            memcpy(&fval, m_buffer + offset, sizeof(fval));
            offset += sizeof(fval);
            w[fid] = fval;
            if (i < SPY_NUM) {
                oss << "<" << fid << "|" << std::fixed << std::setprecision(9) << fval << ">, ";
            }
        }
    } else {
        for (uint32_t i = 0; i < recv_num; ++i) {
            memcpy(&fval, m_buffer + offset, sizeof(fval));
            offset += sizeof(fval);
            w[i] = fval;
            if (i < SPY_NUM) {
                oss << std::fixed << std::setprecision(9) << fval << ", ";
            }
        }
    }
    oss << "\n";
    oss << "[worker][RecvWeightFromMaster] target_num:" << target_num << ", finish\n";
    std::cout << oss.str();
    return true;
}

bool MpiComm::RecvGradientObjFromWorker(const std::vector<WorkerInfo> &workers_info,
        int worker_rank, double *g, double *o, std::vector<uint8_t> &rank_2_part) {
    std::ostringstream oss;
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << "\n";
    int char_num = 0;
    MPI_Recv(m_buffer, kMaxBufferSize, MPI_CHAR, worker_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &m_status);
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << ", recv return\n";
    MPI_Get_count(&m_status, MPI_CHAR, &char_num);
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << ", char_num:" << char_num << "\n";

    int data_part = m_status.MPI_TAG;
    uint32_t target_num = workers_info[data_part].feature_function_num;
    const std::map<uint32_t, std::pair<uint32_t, uint8_t> > &ids_map =
        workers_info[data_part].ids_map;
    rank_2_part[worker_rank] = data_part;
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << ", worker_tag:" << data_part << "\n";

    // check integrity of received data
    uint32_t recv_num = 0;
    memcpy(&recv_num, m_buffer, sizeof(recv_num));
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << ", target_num:" << target_num
        << ", recv_num:" << recv_num << "\n";
    if (recv_num != target_num || (recv_num + 1) * 8 != (uint32_t)(char_num - 4)) return false;
    // read received info
    size_t offset = 4;
    double val = 0.0;
    // 1. obj
    memcpy(&val, m_buffer + offset, sizeof(val));
    offset += sizeof(val);
    *o += val;
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank
        << ", obj:" << std::fixed << std::setprecision(9) << val << ", gradients:";
    // 2. gradient
    for (uint32_t i = 0; i < recv_num; ) {
        std::map<uint32_t, std::pair<uint32_t, uint8_t> >::const_iterator it
            = ids_map.find(i);
        if (ids_map.end() == it) {
            return false;
        } else {
            uint32_t global_id = it->second.first;
            for (size_t j = 0; j < it->second.second; ++j) {
                memcpy(&val, m_buffer + offset, sizeof(val));
                offset += sizeof(val);
                g[global_id + j] += val;
                if (i < SPY_NUM) oss << std::fixed << std::setprecision(9) << val << ", ";
            }
            i += it->second.second;
        }
    }
    oss << "\n";
    oss << "[master][RecvGradientObjFromWorker] worker_rank:" << worker_rank << ", finish\n";
    std::cout << oss.str();
    return true;
}
