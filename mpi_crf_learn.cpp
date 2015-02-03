#include <fstream>
#include <iostream>
#include <iomanip>

#include "common.h"
#include "encoder.h"
#include "lbfgs.h"
#include "mpi.h"
#include "mpi_comm.h"
#include "param.h"

namespace CRFPP {
const Option long_options[] = {
  {"freq",     'f', "1",      "INT",
   "use features that occuer no less than INT(default 1)" },
  {"maxiter" , 'm', "100000", "INT",
   "set INT for max iterations in LBFGS routine(default 10k)" },
  {"cost",     'c', "1.0",    "FLOAT",
   "set FLOAT for cost parameter(default 1.0)" },
  {"eta",      'e', "0.0001", "FLOAT",
   "set FLOAT for termination criterion(default 0.0001)" },
  {"convert",  'C',  0,       0,
   "convert text model to binary model" },
  {"textmodel", 't', 0,       0,
   "build also text model file for debugging" },
  {"algorithm",  'a', "CRF",   "(CRF|MIRA)", "select training algorithm" },
  {"thread", 'p',   "0",       "INT",
   "number of threads (default auto-detect)" },
  {"shrinking-size", 'H', "20", "INT",
   "set INT for number of iterations variable needs to "
   "be optimal before considered for shrinking. (default 20)" },
  {"version",  'v', 0,        0,       "show the version and exit" },
  {"help",     'h', 0,        0,       "show this help and exit" },
  {0, 0, 0, 0, 0}
};
}  // namespace CRFPP

bool loadFeatureIDMap(std::vector<WorkerInfo> &workers_info, uint32_t &total_function_num);

int main(int argc, char *argv[]) {
    CRFPP::Param param;
    param.open(argc, argv, CRFPP::long_options);

    if (!param.help_version()) {
        return 0;
    }

    const bool convert = param.get<bool>("convert");

    const std::vector<std::string> &rest = param.rest_args();
    if (param.get<bool>("help") ||
            (convert && rest.size() != 2) || (!convert && rest.size() != 3)) {
        std::cout << param.help();
        return 0;
    }

    const size_t         freq           = param.get<int>("freq");
    const size_t         maxiter        = param.get<int>("maxiter");
    const double         C              = param.get<float>("cost");
    const double         eta            = param.get<float>("eta");
    const bool           textmodel      = param.get<bool>("textmodel");
    const unsigned short thread         =
        CRFPP::getThreadSize(param.get<unsigned short>("thread"));
    const unsigned short shrinking_size =
        param.get<unsigned short>("shrinking-size");
    std::string salgo = param.get<std::string>("algorithm");

    CRFPP::toLower(&salgo);

    bool orthant = false;
    int algorithm = CRFPP::Encoder::MIRA;
    if (salgo == "crf" || salgo == "crf-l2") {
        algorithm = CRFPP::Encoder::CRF_L2;
        orthant = false;
    } else if (salgo == "crf-l1") {
        algorithm = CRFPP::Encoder::CRF_L1;
        orthant = true;
    } else if (salgo == "mira") {
        algorithm = CRFPP::Encoder::MIRA;
        std::cerr << "MIRA algorithm does NOT support parallelization" << std::endl;
        return -1;
    } else {
        std::cerr << "unknown alogrithm: " << salgo << std::endl;
        return -1;
    }

    char host[MPI_MAX_PROCESSOR_NAME];
    int rank, size, hostlen;
    //MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Get_processor_name(host, &hostlen);
    std::cout << "I'm " << rank << " of " << size << " at " << host;
    if (0 == rank) std::cout << " [master]\n";
    else std::cout << " [worker]\n";

    // create MPI communicator
    MpiComm *comm = new MpiComm();

    if (0 == rank) {  // master
        // load worker's info: feature id map
        uint32_t feature_function_num = 0;
        std::vector<WorkerInfo> workers_info;
        if (!loadFeatureIDMap(workers_info, feature_function_num)) {
            std::cerr << "Load feature id map on master failed" << std::endl;
            goto FAIL_EXIT;
        }
        std::cout << "feature_function_num: " << feature_function_num << "\n";
        // initialize function's parameters & gradients
        std::vector<double> w(feature_function_num, 0.0);
        std::vector<double> g(feature_function_num, 0.0);
        // main control of tranining process
        double old_obj = 1e+37;
        int    converge = 0;
        CRFPP::LBFGS lbfgs;
        int flag = -1;  // 0: success 1: failure
        size_t itr;
        // worker's rank is defined by MPI environment
        // data part is defined by program
        // worker rank i does NOT need to load data part i
        std::vector<uint8_t> rank_2_part(workers_info.size() + 1);
        for (size_t i = 1; i <= workers_info.size(); ++i) {
            // this map will be updated when master receives gradient from workers
            // during training
            rank_2_part[i] = i - 1;
        }
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        for (itr = 0; itr < maxiter; ++itr) {
            double obj = 0.0;
            std::fill(g.begin(), g.end(), 0.0);
            if (0 != itr) {  // gather partial gradient and obj from workers
                for (int i = 1; i < size; ++i) {
                    std::cout << "[master] itr:" << itr << ", recv from worker " << i << " ...\n";
                    comm->RecvGradientObjFromWorker(workers_info, i, &g[0], &obj, rank_2_part);
                    std::cout << "[master] itr:" << itr << ", recv from worker " << i << " complete\n";
                }
                // calc obj diff to determine whether to stop
                double diff = (itr == 0 ? 1.0 :
                        std::abs(old_obj - obj) / old_obj);
                std::cout << "iter="  << itr
                    << " obj=" << obj
                    << " diff="  << diff << std::endl;
                old_obj = obj;

                if (diff < eta) {
                    converge++;
                } else {
                    converge = 0;
                }

                if (converge == 3) {
                    flag = 0;
                    comm->SetFlag();
                    std::cout << "[master] flag:" << comm->GetFlag() << "\n";
                    comm->Bcast();
                    break;  // 3 is ad-hoc
                }
                // update w using lbfgs
                std::cout << "b_lbfgs[g]: ";
                for (size_t i = 0; i < 12; ++i) {
                    std::cout << std::fixed << std::setprecision(9) << g[i] << ',';
                }
                std::cout << std::endl;
                std::cout << "b_lbfgs[w]: ";
                for (size_t i = 0; i < 12; ++i) {
                    std::cout << std::fixed << std::setprecision(9) << w[i] << ',';
                }
                std::cout << std::endl;
                int ret = lbfgs.optimize(feature_function_num,
                        &w[0],
                        obj,
                        &g[0], orthant, C);
                std::cout << "a_lbfgs[w]: ";
                for (size_t i = 0; i < 12; ++i) {
                    std::cout << std::fixed << std::setprecision(9) << w[i] << ',';
                }
                std::cout << std::endl;
                if (ret <= 0) {
                    flag = 1;
                    comm->SetFlag();
                    std::cout << "[master] flag:" << comm->GetFlag() << "\n";
                    comm->Bcast();
                    break;
                }
            }
            comm->Bcast();
            // send weight to workers
            for (int i = 1; i < size; ++i) {
                std::cout << "[master] itr:" << itr << ", send to worker " << i << " ...\n";
                comm->SendWeightToWorker(&w[0], workers_info[rank_2_part[i]], i, orthant);
                std::cout << "[master] itr:" << itr << ", send to worker " << i << " complete\n";
            }
        }
        std::cout << "[master] itr:" << itr << ", maxiter:" << maxiter << ", flag:" << flag << "\n";
        if (0 == flag || itr >= maxiter) {  // success: output parameters
//            for (size_t i = 0; i < w.size(); ++i) {
//                std::cout << w[i] << std::endl;
//            }
            goto SUCC_EXIT;
        } else {
            goto FAIL_EXIT;
        }
    } else {  // worker
        // TODO: fix me
        std::vector<std::string> y;
        y.push_back("B");
        y.push_back("E");
        y.push_back("I");
        y.push_back("O");
        CRFPP::Encoder encoder;
        encoder.setMpiComm(comm);
        if (!encoder.learn(rest[0].c_str(),
                    rest[1].c_str(),
                    rest[2].c_str(),
                    textmodel,
                    maxiter, freq, eta, C, thread, shrinking_size,
                    algorithm, y)) {
            std::cerr << encoder.what() << std::endl;
            goto FAIL_EXIT;
        } else {
            goto SUCC_EXIT;
        }
    }

FAIL_EXIT:
    delete comm;
    MPI_Finalize();
    std::cout << "[main] fail exit\n";
    return -1;

SUCC_EXIT:
    delete comm;
    MPI_Finalize();
    std::cout << "[main] succ exit\n";
    return 0;
}

bool loadFeatureIDMap(std::vector<WorkerInfo> &workers_info, uint32_t &total_function_num) {
    std::ifstream ifs("/data/share/johnxuan/crf/features_map");
    if (!ifs.is_open()) {
        std::cerr << "Open feature id map failed" << std::endl;
        return false;
    }

    workers_info.clear();
    total_function_num = 0;

    WorkerInfo worker_info;
    uint32_t id, local_id, global_id, function_num;
    uint32_t last_id(0), worker_function_num(0), line_num(0);
    std::map<uint32_t, std::pair<uint32_t, uint8_t> > info;
    std::map<uint32_t, uint8_t> function_info;
    while (ifs >> id >> local_id >> global_id >> function_num) {
        ++line_num;
        if (id != last_id) {  // new worker
            // record last worker's info
            worker_info.data_part_id = last_id;
            worker_info.feature_function_num = worker_function_num;
            worker_info.ids_map = info;
            if (last_id >= workers_info.size()) {
                workers_info.resize(last_id + 1);
            }
            workers_info[last_id] = worker_info;
            // update
            last_id = id;
            worker_function_num = function_num;
            info.clear();
            info.insert(std::make_pair(local_id, std::make_pair(global_id, function_num)));
        } else {  // current worker
            worker_function_num += function_num;
            info.insert(std::make_pair(local_id, std::make_pair(global_id, function_num)));
        }

        std::map<uint32_t, uint8_t>::iterator it = function_info.find(global_id);
        if (function_info.end() == it) {
            function_info.insert(std::make_pair(global_id, function_num));
        }
    }
    ifs.close();
    
    // record the last one's info
    worker_info.data_part_id = last_id;
    worker_info.feature_function_num = worker_function_num;
    worker_info.ids_map = info;
    if (last_id >= workers_info.size()) {
        workers_info.resize(last_id + 1);
    }
    workers_info[last_id] = worker_info;

    // calculate feature function's num
    for (std::map<uint32_t, uint8_t>::iterator it = function_info.begin();
            it != function_info.end(); ++it) {
        total_function_num += it->second;
    }

    return true;
}
