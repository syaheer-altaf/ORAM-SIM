/*
** This experiment finds the threshold for setting the bucket sizes of lower levels of the
** Path ORAM tree to Z=1 while the rest is still 4.
** For height L, let
** Z(d) = 4 if 0 <= d <= L-k
** Z(d) = 0 if L-k+1 <= d <= L,
** where d refers to depth and k is a fixed integer.
*/

#include "../lp_oram.cpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
namespace fs = std::filesystem;

const std::uint64_t T_warm = 1000000;           // 10^6 accesses
const std::uint64_t T_meas = 1000000;           // 10^6 accesses

void run_experiemnt(ORAM& o, std::ofstream& csv, u32 L, u32 k) {    
    u32 idx = 0;
    /* Warm-up phase */
    for (std::uint64_t t = 0; t < T_warm; ++t) {
        access(o, idx % (1u << L));
        idx++;
    }
    /* Measurement phase -- measure stash */
    // Compute stats -- variance via Welford's method
    std::uint64_t n = 0;
    double mean_stash = 0.0, M2 = 0.0;
    for (std::uint64_t t = 0; t < T_meas; ++t) {
        access(o, idx % (1u << L));
        idx++;
        const double s = o.stash.size();
        o.max_stash = std::max(o.max_stash, o.stash.size());
        ++n;
        const double delta = s - mean_stash;
        mean_stash += delta / n;
        M2 += delta * (s - mean_stash);
    }
    const double std_stash = std::sqrt(M2 / n);

    csv << L << ',' << k << ',' << T_warm << ',' << T_meas << ','
        << o.max_stash << ',' << mean_stash << ',' << std_stash << ','
        << (check_invariant(o) ? "OK" : "BROKEN") << '\n' << std::flush;
}

int main() {
    fs::path dir_path = "./results/exp1";

    try {
        if (fs::create_directories(dir_path)) {
            std::cout << "Directory created successfully.\n";
        } else {
            std::cout << "Directory already exists or could not be created.\n";
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
    const fs::path csv_path = dir_path / "exp1.csv";
    const bool need_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) throw std::runtime_error("Could not open exp1.csv for writing");

    csv << std::setprecision(10);

    if (need_header)
        csv << "L,k,T_warm,T_meas,max_stash,mean_stash,std_stash,invariant\n";
    std::vector<u32> heights = {15,16,17,18,19,20};
    std::vector<u32> thres_k = {0,1,2,3,4};
    for (const auto& L : heights) {
        for (const auto& k : thres_k) {
            ORAM o; init(o, L, [L, k](u32 level) {return level <= L-k ? 4u : 0u;});
            std::cout << "init done, invariant " << (check_invariant(o) ? "OK" : "BROKEN") << ", stash " << o.stash.size() << "\n";
            run_experiemnt(o, csv, L, k);
        }
    }
    return 0;
}