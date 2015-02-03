#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

bool GetCorpusInfo(const char *file, uint32_t &total_len,
        uint32_t &line_num, uint32_t &sent_num);
bool SplitCorpus(const char *file, int part_num,
        uint32_t avg_len_per_part, uint32_t avg_len_per_sent);

int main(int argc, char *argv[]) {
    if (3 != argc) {
        std::cerr << "usage: " << argv[0] << " partition_number corpus_file\n";
        return -1;
    }

    uint32_t total_len(0), line_num(0), sent_num(0);
    if (!GetCorpusInfo(argv[2], total_len, line_num, sent_num)) {
        std::cerr << "Get corpus [" << argv[2] << "] info failed!\n";
        return -1;
    }

    int p_num = atoi(argv[1]);
    int avg_len_per_part = total_len / p_num;
    int avg_len_per_sent = total_len / sent_num;
    if (!SplitCorpus(argv[2], p_num, avg_len_per_part, avg_len_per_sent)) {
        std::cerr << "Split corpus [" << argv[2] << "] failed!\n";
        return -1;
    }

    return 0;
}

bool GetCorpusInfo(const char *file, uint32_t &total_len,
        uint32_t &line_num, uint32_t &sent_num) {
    total_len = line_num = sent_num = 0;

    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        std::cerr << "Open file [" << file << "] failed!\n";
        return false;
    }
    std::string line;
    while (getline(ifs, line)) {
        ++line_num;
        if (line[0] == '\0' || line[0] == ' ' || line[0] == '\t') {
            ++sent_num;
            continue;
        }
        size_t pos = line.find_first_of("\t ");
        total_len += pos;
    }
    ifs.close();

    return true;
}

bool SplitCorpus(const char *file, int part_num,
        uint32_t avg_len_per_part, uint32_t avg_len_per_sent) {
    // construct file name for partitions
    std::ostringstream oss;
    std::vector<std::string> p_file_name(part_num);
    for (int i = 0; i < part_num; ++i) {
        oss.str("");
        oss << file << "_" << i;
        p_file_name[i] = oss.str();
    }
    // construct ofstream for partitions
    std::vector<std::ofstream *> ofs(part_num);
    for (int i = 0; i < part_num; ++i) {
        std::ofstream *o = new std::ofstream(p_file_name[i].c_str());
        ofs[i] = o;
    }

    // split corpus
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        std::cerr << "Open file [" << file << "] failed!\n";
        return false;
    }
    int pid = 0;
    uint32_t total_len = 0;
    std::string line;
    std::vector<std::string> sent_info;
    while (getline(ifs, line)) {
        sent_info.push_back(line);
        if (line[0] == '\0' || line[0] == ' ' || line[0] == '\t') {
            size_t sent_len = 0;
            for (size_t i = 0; i < sent_info.size(); ++i) {
                size_t pos = sent_info[i].find_first_of("\t ");
                if (std::string::npos == pos) pos = 0;
                sent_len += pos;
            }
            int next = pid + 1;
            if (next >= part_num) next = part_num - 1;
            int belong = -1;
            if (total_len >= avg_len_per_part) {
                belong = next;
                total_len = sent_len;
                ++pid;
            } else if (total_len + sent_len <= avg_len_per_part
                    || total_len + sent_len <= avg_len_per_part + avg_len_per_sent) {
                belong = pid;
                total_len += sent_len;
            } else {
                belong = next;
                total_len = sent_len;
                ++pid;
            }
            if (pid >= part_num) pid = part_num - 1;
            for (size_t i = 0; i < sent_info.size(); ++i) {
                (*ofs[belong]) << sent_info[i] << std::endl;
            }
            sent_info.clear();
        }
    }
    ifs.close();

    for (int i = 0; i < part_num; ++i) {
        ofs[i]->close();
        delete ofs[i];
    }

    return true;
}
