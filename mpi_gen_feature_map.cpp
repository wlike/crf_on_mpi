#include <fstream>
#include <iostream>
#include <string>

#include "common.h"
#include "feature_index.h"
#include "tagger.h"

using namespace CRFPP;

// output file format:
//   part_id \t local_feature_id \t global_feature_id \t feature_function_num

bool readFile(const char *template_file, const char *train_file,
        EncoderFeatureIndex *index, bool is_part);
void genFeatureIDMap(std::ofstream *ofs, int part_id, int cls_num,
        const std::map<std::string, FeatureInfo> &total,
        const std::map<std::string, FeatureInfo> &part);

std::vector<std::string> y;

int main(int argc, char *argv[]) {
    if (5 != argc) {
        std::cerr << "usage: " << argv[0] << " template_file train_file part_num res_file\n";
        return -1;
    }

    EncoderFeatureIndex feature_index, feature_index_part;

    // read total train file
    if (!readFile(argv[1], argv[2], &feature_index, false)) {
        std::cerr << "Read file [" << argv[1] << "] failed!\n";
        return -1;
    }
    const std::map<std::string, FeatureInfo> &features =
        feature_index.getFeatureIndex();
    y = feature_index.getY();

    std::ofstream ofs(argv[4]);
    // read each part train file
    std::ostringstream oss;
    for (int i = 0; i < atoi(argv[3]); ++i) {
        oss.str("");
        oss << argv[2] << '_' << i;
        std::string file_name = oss.str();
        feature_index_part.clear();
        if (!readFile(argv[1], file_name.c_str(), &feature_index_part, true)) {
            std::cerr << "Read file [" << file_name << "] failed!\n";
            return -1;
        }
        genFeatureIDMap(&ofs, i, feature_index.ysize(), features,
                feature_index_part.getFeatureIndex());
    }
    ofs.close();

    std::cout << "Total feature function number: " << feature_index.size() << std::endl;

    return 0;
}

bool readFile(const char *template_file, const char *train_file,
        EncoderFeatureIndex *feature_index, bool is_part) {
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

    // correct y of partial corpus
    if (is_part) {
        feature_index->setY(y);
    }

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

void genFeatureIDMap(std::ofstream *ofs, int part_id, int cls_num,
        const std::map<std::string, FeatureInfo> &total,
        const std::map<std::string, FeatureInfo> &part) {
    typedef std::map<std::string, FeatureInfo>::const_iterator map_cit;

    int feature_function_num = 0;
    for (map_cit it = part.begin(); it != part.end(); ++it) {
        const char *feature = it->first.c_str();
        if ('U' == feature[0]) feature_function_num = cls_num;
        else if ('B' == feature[0]) feature_function_num = cls_num * cls_num;
        map_cit in_it = total.find(it->first);
        (*ofs) << part_id << '\t' << it->second.id_ << '\t'
            << in_it->second.id_ << '\t' << feature_function_num << std::endl;
    }
}
