#include <fstream>
#include <iostream>

#include "common.h"
#include "feature_index.h"
#include "tagger.h"

using namespace CRFPP;

bool readFile(const char *template_file, const char *train_file,
        EncoderFeatureIndex *index);

int main(int argc, char *argv[]) {
    if (5 != argc) {
        std::cerr << "usage: " << argv[0] << " template_file train_file param_file model_file\n";
        return -1;
    }

    EncoderFeatureIndex feature_index;

    // read template/train file
    if (!readFile(argv[1], argv[2], &feature_index)) {
        std::cerr << "Read file [" << argv[1] << "] failed!\n";
        return -1;
    }

    // read param file
    std::ifstream ifs(argv[3]);
    if (!ifs.is_open()) {
        std::cerr << "Open param file [" << argv[3] << "] failed!\n";
        return -1;
    }
    double param = 0.0;
    std::vector<double> alpha;
    while (ifs >> param) {
        alpha.push_back(param);
    }
    ifs.close();

    feature_index.set_alpha(&alpha[0]);

    if (!feature_index.save(argv[4], true)) {
        std::cerr << "Save assembled crf model failed!\n";
        return -1;
    }

    return 0;
}

bool readFile(const char *template_file, const char *train_file,
        EncoderFeatureIndex *feature_index) {
    Allocator allocator;
    std::vector<TaggerImpl* > x;
    whatlog what_;  // for CHECK_FALSE

#define WHAT_ERROR(msg) do {                                    \
    for (std::vector<TaggerImpl *>::iterator it = x.begin();    \
            it != x.end(); ++it)                                \
        delete *it;                                             \
    std::cerr << msg << std::endl;                              \
    return false; } while (0)

    CHECK_FALSE(feature_index->open(template_file, train_file))
        << feature_index->what();

    {
        std::ifstream ifs(train_file);
        CHECK_FALSE(ifs) << "cannot open: " << train_file;

        while (ifs) {
            TaggerImpl *_x = new TaggerImpl();
            _x->open(feature_index, &allocator);
            if (!_x->read(&ifs) || !_x->shrink()) {
                WHAT_ERROR(_x->what());
            }

            if (!_x->empty()) {
                x.push_back(_x);
            } else {
                delete _x;
                continue;
            }
        }

        ifs.close();
    }

    for (std::vector<TaggerImpl *>::iterator it = x.begin();
            it != x.end(); ++it) {
        delete *it;
    }

    return true;
}
