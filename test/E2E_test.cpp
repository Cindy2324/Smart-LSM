#include "../test.h"
#include <fstream>
#include <iostream>
#include <string>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>


std::vector<std::string> read_file(std::string filename) {
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr<<"Failed to open file: "<<filename<<std::endl;
        return {};
    }
    std::string line;
    std::vector<std::string> temp;
    while (std::getline(file, line)) {
        bool exist_alpha = false;
        for (auto c : line) {
            if (isalpha(c)) {
                exist_alpha = true;
                break;
            }
        }
        if (!exist_alpha) {
            continue;
        }
        if (line.empty())
            continue;
        if(line.size() < 70) {
            continue;
        }
        temp.push_back(line);
    }
    file.close();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "File reading time: " << duration.count() << " seconds." << std::endl;
    return temp;
}

class CorrectnessTest : public Test {
private:
    const uint64_t SIMPLE_TEST_MAX = 512;
    const uint64_t MIDDLE_TEST_MAX  = 1024 * 64;
    const uint64_t LARGE_TEST_MAX  = 1024 * 64;

	void text_test(uint64_t max) {
		auto start_total = std::chrono::high_resolution_clock::now();

		uint64_t i;
		auto trimmed_text = read_file("./data/trimmed_text.txt");
		max				  = std::min(max, (uint64_t)trimmed_text.size());

		auto start_storage = std::chrono::high_resolution_clock::now();
		for (i = 0; i < max; ++i) {
			store.put(i, trimmed_text[i]);
		}
		auto end_storage = std::chrono::high_resolution_clock::now();
		std::cout << "Data storage time: " << std::chrono::duration<double>(end_storage - start_storage).count() << " seconds." << std::endl;

		auto start_retrieval = std::chrono::high_resolution_clock::now();
		for (i = 0; i < max; ++i)
			EXPECT(trimmed_text[i], store.get(i));
		// phase();
		auto end_retrieval = std::chrono::high_resolution_clock::now();
		std::cout << "Data retrieval time: " << std::chrono::duration<double>(end_retrieval - start_retrieval).count() << " seconds." << std::endl;

		// run the search_knn, and compare the result to ./data/test_text_ans.txt
		auto test_text = read_file("./data/test_text.txt");
		max			   = std::min(max, (uint64_t)test_text.size());

		std::vector<std::string> ans;
        ans = read_file("./data/test_text_ans.txt");
        phase();
		auto start_knn = std::chrono::high_resolution_clock::now();
		int idx = 0, k = 3;
		for (i = 0; i < max; ++i) {
			auto res = store.search_knn(test_text[i], k);
			for (auto j : res) {
                if(store.get(j.first) != j.second) {
                    std::cerr << "TEST Error @" << __FILE__ << ":" << __LINE__;
                    std::cerr << ", expected " << ans[idx];
                    std::cerr << ", got " << j.second << std::endl;
                }
				EXPECT(ans[idx], j.second);
				idx++;
			}
		}
		auto end_knn = std::chrono::high_resolution_clock::now();
		std::cout << "KNN search time: " << std::chrono::duration<double>(end_knn - start_knn).count() << " seconds." << std::endl;

		auto phase_with_tolerance = [this](double tolerance = 0.03) {
			// Report
			std::cout << "  Phase " << (nr_phases + 1) << ": ";
			std::cout << nr_passed_tests << "/" << nr_tests << " ";

			// Calculate tolerance
			double pass_rate		   = static_cast<double>(nr_passed_tests) / nr_tests;
			bool passed_with_tolerance = pass_rate >= (1.0 - tolerance);

			// Count
			++nr_phases;
			if (passed_with_tolerance) {
				++nr_passed_phases;
				std::cout << "[PASS]" << std::endl;
			} else {
                std::cout << "Accept Rate: " << pass_rate * 100 << "%\n";
                std::cout << "The Accept Rate we recommend is more than 85%.\nBecause the embedding model may not act strictly samely between each machine.\n";
			}

			std::cout.flush();

			// Reset
			nr_tests		= 0;
			nr_passed_tests = 0;
		};
		phase_with_tolerance(0.15);
		auto end_total = std::chrono::high_resolution_clock::now();
		std::cout << "Total text test time: " << std::chrono::duration<double>(end_total - start_total).count() << " seconds." << std::endl;
	}

public:
    CorrectnessTest(const std::string &dir, bool v = true) : Test(dir, v) {}

    void start_test(void *args = NULL) override {
        std::cout << "===========================" << std::endl;
        std::cout << "KVStore Correctness Test" << std::endl;

    	auto start_reset = std::chrono::high_resolution_clock::now();
        store.reset();
    	auto end_reset = std::chrono::high_resolution_clock::now();
    	std::cout << "KVStore reset time: " << std::chrono::duration<double>(end_reset - start_reset).count() << " seconds." << std::endl;

        std::cout << "[Text Test]" << std::endl;
        text_test(120);
    }
};

int main(int argc, char *argv[]) {
    bool verbose = (argc == 2 && std::string(argv[1]) == "-v");

    std::cout << "Usage: " << argv[0] << " [-v]" << std::endl;
    std::cout << "  -v: print extra info for failed tests [currently ";
    std::cout << (verbose ? "ON" : "OFF") << "]" << std::endl;
    std::cout << std::endl;
    std::cout.flush();

	auto start_init = std::chrono::high_resolution_clock::now();
    CorrectnessTest test("./data", verbose);
	auto end_init = std::chrono::high_resolution_clock::now();
	std::cout << "KVStore init time: " << std::chrono::duration<double>(end_init - start_init).count() << " seconds." << std::endl;

    test.start_test();

    return 0;
}