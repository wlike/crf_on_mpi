#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// make continuous blank lines to be one line in corpus
int main(int argc, char *argv[]) {
    if (3 != argc) {
        std::cerr << "usage: " << argv[0] << " orig_corpus_file new_corpus_file\n";
        return -1;
    }

    std::ifstream ifs(argv[1]);
    if (!ifs.is_open()) {
        std::cerr << "Open orig corpus file [" << argv[1] << "] failed!\n";
        return -1;
    }

    std::string line;
    std::vector<std::string> blanks;
    std::ofstream ofs(argv[2]);
    while (getline(ifs, line)) {
        if (line[0] == '\0') {
            if (blanks.size() == 0) {  // first blank line
                ofs << line << std::endl;
                blanks.push_back(line);
            }
        } else {
            blanks.clear();
            ofs << line << std::endl;
        }
    }
    ifs.close();
    ofs.close();

    return 0;
}
