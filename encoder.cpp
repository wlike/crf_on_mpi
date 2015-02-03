//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.cpp 1601 2007-03-31 09:47:18Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <algorithm>
#include <fstream>
#include "param.h"
#include "encoder.h"
#include "timer.h"
#include "tagger.h"
#include "lbfgs.h"
#include "common.h"
#include "feature_index.h"
#include "scoped_ptr.h"
#include "thread.h"

namespace CRFPP {

class CRFEncoderThread: public thread {
 public:
  TaggerImpl **x;
  unsigned short start_i;  // start sentence id assigned to this thread
  unsigned short thread_num;
  int zeroone;
  int err;
  size_t size;  // total sentence number
  double obj;
  std::vector<double> expected;

  void run() {
    obj = 0.0;
    err = zeroone = 0;
    std::fill(expected.begin(), expected.end(), 0.0);
    for (size_t i = start_i; i < size; i += thread_num) {
      obj += x[i]->gradient(&expected[0]);
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) {
        ++zeroone;
      }
    }
  }
};

bool runMIRA(const std::vector<TaggerImpl* > &x,
             EncoderFeatureIndex *feature_index,
             double *alpha,
             size_t maxitr,
             float C,
             double eta,
             unsigned short shrinking_size,
             unsigned short thread_num) {
  std::vector<unsigned char> shrink(x.size());
  std::vector<float> upper_bound(x.size());
  std::vector<double> expected(feature_index->size());

  std::fill(upper_bound.begin(), upper_bound.end(), 0.0);
  std::fill(shrink.begin(), shrink.end(), 0);

  int converge = 0;
  int all = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    all += x[i]->size();
  }

  for (size_t itr = 0; itr < maxitr; ++itr) {
    int zeroone = 0;
    int err = 0;
    int active_set = 0;
    int upper_active_set = 0;
    double max_kkt_violation = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
      if (shrink[i] >= shrinking_size) {
        continue;
      }

      ++active_set;
      std::fill(expected.begin(), expected.end(), 0.0);
      double cost_diff = x[i]->collins(&expected[0]);
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) {
        ++zeroone;
      }

      if (error_num == 0) {
        ++shrink[i];
      } else {
        shrink[i] = 0;
        double s = 0.0;
        for (size_t k = 0; k < expected.size(); ++k) {
          s += expected[k] * expected[k];
        }

        double mu = std::max(0.0, (error_num - cost_diff) / s);

        if (upper_bound[i] + mu > C) {
          mu = C - upper_bound[i];
          ++upper_active_set;
        } else {
          max_kkt_violation = std::max(error_num - cost_diff,
                                   max_kkt_violation);
        }

        if (mu > 1e-10) {
          upper_bound[i] += mu;
          upper_bound[i] = std::min(C, upper_bound[i]);
          for (size_t k = 0; k < expected.size(); ++k) {
            alpha[k] += mu * expected[k];
          }
        }
      }
    }

    double obj = 0.0;
    for (size_t i = 0; i < feature_index->size(); ++i) {
      obj += alpha[i] * alpha[i];
    }

    std::cout << "iter="  << itr
              << " terr=" << 1.0 * err / all
              << " serr=" << 1.0 * zeroone / x.size()
              << " act=" <<  active_set
              << " uact=" << upper_active_set
              << " obj=" << obj
              << " kkt=" << max_kkt_violation << std::endl;

    if (max_kkt_violation <= 0.0) {
      std::fill(shrink.begin(), shrink.end(), 0);
      converge++;
    } else {
      converge = 0;
    }

    if (itr > maxitr || converge == 2) {
      break;  // 2 is ad-hoc
    }
  }

  return true;
}

bool runCRF(const std::vector<TaggerImpl* > &x,
            EncoderFeatureIndex *feature_index,
            double *alpha,
            size_t maxitr,
            float C,
            double eta,
            unsigned short thread_num,
            bool orthant
#ifdef USE_MPI
            , MpiComm *comm
            , uint8_t data_part
#endif  // USE_MPI
            ) {
#ifndef USE_MPI
  double old_obj = 1e+37;
  int    converge = 0;
  LBFGS lbfgs;
#endif  // USE_MPI
  std::vector<CRFEncoderThread> thread(thread_num);

  for (size_t i = 0; i < thread_num; i++) {
    thread[i].start_i = i;
    thread[i].size = x.size();
    thread[i].thread_num = thread_num;
    thread[i].x = const_cast<TaggerImpl **>(&x[0]);
    thread[i].expected.resize(feature_index->size());
  }

  size_t all = 0;  // tags/tokens number
  for (size_t i = 0; i < x.size(); ++i) {
    all += x[i]->size();
  }

  for (size_t itr = 0; itr < maxitr; ++itr) {
#ifdef USE_MPI
    comm->Bcast();
    std::cout << "[worker] itr:" << itr << ", flag:" << comm->GetFlag() << "\n";
    if (1 == comm->GetFlag()) break;
    std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", recv weight from master ...\n";
    if (!comm->RecvWeightFromMaster(orthant, feature_index->size(),
                feature_index->alpha())) {
      std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", recv weight from master failed\n";
      return false;
    }
    std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", recv weight from master complete\n";
#endif  // USE_MPI
    for (size_t i = 0; i < thread_num; ++i) {
      thread[i].start();
    }

    for (size_t i = 0; i < thread_num; ++i) {
      thread[i].join();
    }

    for (size_t i = 1; i < thread_num; ++i) {
      thread[0].obj += thread[i].obj;
      thread[0].err += thread[i].err;
      thread[0].zeroone += thread[i].zeroone;
    }

    for (size_t i = 1; i < thread_num; ++i) {
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].expected[k] += thread[i].expected[k];
      }
    }

    size_t num_nonzero = 0;
    if (orthant) {  // L1 regularization
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += std::abs(alpha[k] / C);
        if (alpha[k] != 0.0) {
          ++num_nonzero;
        }
      }
    } else {  // L2 regularization
      num_nonzero = feature_index->size();
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += (alpha[k] * alpha[k] / (2.0 * C));
        thread[0].expected[k] += alpha[k] / C;
      }
    }

#ifndef USE_MPI
    double diff = (itr == 0 ? 1.0 :
                   std::abs(old_obj - thread[0].obj) / old_obj);
    std::cout << "iter="  << itr
              << " terr=" << 1.0 * thread[0].err / all
              << " serr=" << 1.0 * thread[0].zeroone / x.size()
              << " act=" << num_nonzero
              << " obj=" << thread[0].obj
              << " diff="  << diff << std::endl;
    old_obj = thread[0].obj;

    if (diff < eta) {
      converge++;
    } else {
      converge = 0;
    }

    if (itr > maxitr || converge == 3) {
      break;  // 3 is ad-hoc
    }

    std::cout << "b_lbfgs[g]: ";
    for (size_t i = 0; i < 10 && i < thread[0].expected.size(); ++i) {
        std::cout << thread[0].expected[i] << ',';
    }
    std::cout << std::endl;
    std::cout << "b_lbfgs[w]: ";
    const double *b_alpha = feature_index->alpha();
    for (size_t i = 0; i < 10 && i < feature_index->size(); ++i) {
        std::cout << b_alpha[i] << ',';
    }
    std::cout << std::endl;
    int ret = lbfgs.optimize(feature_index->size(),
                       &alpha[0],
                       thread[0].obj,
                       &thread[0].expected[0], orthant, C);
    std::cout << "a_lbfgs[w]: ";
    const double *a_alpha = feature_index->alpha();
    for (size_t i = 0; i < 10 && i < feature_index->size(); ++i) {
        std::cout << a_alpha[i] << ',';
    }
    std::cout << std::endl;
    if (ret <= 0) {
      return false;
    }
#else
    std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", send to master ...\n";
    if (!comm->SendGradientObjToMaster(&thread[0].expected[0],
                feature_index->size(), thread[0].obj, data_part)) {
      std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", send to master failed\n";
      return false;
    }
    std::cout << "[worker] itr:" << itr << ", data_part:" << (int)data_part << ", send to master complete\n";
#endif  // USE_MPI
  }

  return true;
}

bool Encoder::convert(const char *textfilename,
                      const char *binaryfilename) {
  EncoderFeatureIndex feature_index;
  CHECK_FALSE(feature_index.convert(textfilename, binaryfilename))
      << feature_index.what();

  return true;
}

bool Encoder::learn(const char *templfile,
                    const char *trainfile,
                    const char *modelfile,
                    bool textmodelfile,
                    size_t maxitr,
                    size_t freq,
                    double eta,
                    double C,
                    unsigned short thread_num,
                    unsigned short shrinking_size,
                    int algorithm
#ifdef USE_MPI
                    , const std::vector<std::string> &y
#endif  // USE_MPI
                    ) {
  std::cout << COPYRIGHT << std::endl;

  CHECK_FALSE(eta > 0.0) << "eta must be > 0.0";
  CHECK_FALSE(C >= 0.0) << "C must be >= 0.0";
  CHECK_FALSE(shrinking_size >= 1) << "shrinking-size must be >= 1";
  CHECK_FALSE(thread_num > 0) << "thread must be > 0";

#ifndef CRFPP_USE_THREAD
  CHECK_FALSE(thread_num == 1)
      << "This architecture doesn't support multi-thrading";
#endif

  if (algorithm == MIRA && thread_num > 1) {
    std::cerr << "MIRA doesn't support multi-thrading. use thread_num=1"
              << std::endl;
  }

  EncoderFeatureIndex feature_index;
  Allocator allocator(thread_num);
  std::vector<TaggerImpl* > x;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(5);

#define WHAT_ERROR(msg) do {                                    \
    for (std::vector<TaggerImpl *>::iterator it = x.begin();    \
         it != x.end(); ++it)                                   \
      delete *it;                                               \
    std::cerr << msg << std::endl;                              \
    return false; } while (0)

  CHECK_FALSE(feature_index.open(templfile, trainfile))
      << feature_index.what();

#ifdef USE_MPI
  feature_index.setY(y);
#endif  // USE_MPI

  {
    progress_timer pg;

#ifdef USE_MPI
    // format: xxx_id
    std::string str_trainfile(trainfile);
    size_t pos = str_trainfile.find('_');
    data_part_ = atoi(str_trainfile.substr(pos+1).c_str());
#endif  // USE_MPI

    std::ifstream ifs(WPATH(trainfile));
    CHECK_FALSE(ifs) << "cannot open: " << trainfile;

    std::cout << "reading training data: " << std::flush;
    size_t line = 0;
    while (ifs) {
      TaggerImpl *_x = new TaggerImpl();
      _x->open(&feature_index, &allocator);
      if (!_x->read(&ifs) || !_x->shrink()) {
        WHAT_ERROR(_x->what());
      }

      if (!_x->empty()) {
        x.push_back(_x);
      } else {
        delete _x;
        continue;
      }

      _x->set_thread_id(line % thread_num);

      if (++line % 100 == 0) {
        std::cout << line << "... " << std::flush;
      }
    }

    ifs.close();
    std::cout << "\nDone!";
  }

  feature_index.shrink(freq, &allocator);

  std::vector<double> alpha(feature_index.size());  // parameters
  std::fill(alpha.begin(), alpha.end(), 0.0);
  feature_index.set_alpha(&alpha[0]);

  std::cout << "Number of sentences:         " << x.size() << std::endl;
  std::cout << "Number of feature functions: " << feature_index.size() << std::endl;
  std::cout << "Number of thread(s):         " << thread_num << std::endl;
  std::cout << "Freq:                        " << freq << std::endl;
  std::cout << "eta:                         " << eta << std::endl;
  std::cout << "C:                           " << C << std::endl;
  std::cout << "shrinking size:              " << shrinking_size
            << std::endl;

  progress_timer pg;

  switch (algorithm) {
    case MIRA:
      if (!runMIRA(x, &feature_index, &alpha[0],
                   maxitr, C, eta, shrinking_size, thread_num)) {
        WHAT_ERROR("MIRA execute error");
      }
      break;
    case CRF_L2:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, thread_num, false
#ifdef USE_MPI
                  , comm_, data_part_
#endif  // USE_MPI
                  )) {
        WHAT_ERROR("CRF_L2 execute error");
      }
      break;
    case CRF_L1:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, thread_num, true
#ifdef USE_MPI
                  , comm_, data_part_
#endif  // USE_MPI
                  )) {
        WHAT_ERROR("CRF_L1 execute error");
      }
      break;
  }

  for (std::vector<TaggerImpl *>::iterator it = x.begin();
       it != x.end(); ++it) {
    delete *it;
  }

#ifndef USE_MPI
  if (!feature_index.save(modelfile, textmodelfile)) {
    WHAT_ERROR(feature_index.what());
  }
#endif  // USE_MPI

  std::cout << "\nDone!";

  return true;
}

namespace {
const CRFPP::Option long_options[] = {
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
   "be optimal before considered for shrinking in MIRA. (default 20)" },
  {"version",  'v', 0,        0,       "show the version and exit" },
  {"help",     'h', 0,        0,       "show this help and exit" },
  {0, 0, 0, 0, 0}
};

#ifndef USE_MPI
int crfpp_learn(const Param &param) {
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

  int algorithm = CRFPP::Encoder::MIRA;
  if (salgo == "crf" || salgo == "crf-l2") {
    algorithm = CRFPP::Encoder::CRF_L2;
  } else if (salgo == "crf-l1") {
    algorithm = CRFPP::Encoder::CRF_L1;
  } else if (salgo == "mira") {
    algorithm = CRFPP::Encoder::MIRA;
  } else {
    std::cerr << "unknown alogrithm: " << salgo << std::endl;
    return -1;
  }

  CRFPP::Encoder encoder;
  if (convert) {
    if (!encoder.convert(rest[0].c_str(), rest[1].c_str())) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  } else {
    if (!encoder.learn(rest[0].c_str(),
                       rest[1].c_str(),
                       rest[2].c_str(),
                       textmodel,
                       maxiter, freq, eta, C, thread, shrinking_size,
                       algorithm)) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  }

  return 0;
}
#endif  // USE_MPI

}  // namespace
}  // namespace CRFPP

#ifndef USE_MPI
int crfpp_learn2(const char *argv) {
  CRFPP::Param param;
  param.open(argv, CRFPP::long_options);
  return CRFPP::crfpp_learn(param);
}

int crfpp_learn(int argc, char **argv) {
  CRFPP::Param param;
  param.open(argc, argv, CRFPP::long_options);
  return CRFPP::crfpp_learn(param);
}
#endif  // USE_MPI

