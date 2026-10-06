#include "../lp_oram.cpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

const std::uint64_t T_warm = 100000000;           // 10^8 accesses
const std::uint64_t T_meas = 100000000;           // 10^8 accesses

void print_csv(std::ofstream& csv, std::ofstream& hcsv, u32 L, u32 k, u32 m, 
    std::vector<std::uint64_t> hist, u32 max_stash, double mean_stash, double std_stash, bool check_inv) {
    csv << L << ',' << k << ',' << m << ','
        << max_stash << ',' << mean_stash << ',' << std_stash << ','
        << (check_inv ? "OK" : "BROKEN") << '\n' << std::flush;

    for (std::size_t s = 0; s < hist.size(); ++s)
        if (hist[s])
            hcsv << L << ',' << k << ',' << m << ',' << s << ',' << hist[s] << '\n';
    hcsv << std::flush;
}

void run_experiment(ORAM& o, std::ofstream& csv, std::ofstream& hcsv, u32 L, u32 k, u32 m) {
    // to ensure fair comparisons
    std::uint64_t new_T_warm = ceil((double) T_warm / m);
    std::uint64_t new_T_meas = ceil((double) T_meas / m);

    u32 idx = 0;

    auto next_id = [L, &idx]() {
        u32 i = idx % (1u << L);
        idx++;
        return i;
    };
    /* Warm-up phase */
    for (std::uint64_t t = 0; t < new_T_warm; ++t) {
        std::vector<u32> ids(m);
        std::generate(ids.begin(), ids.end(), next_id);
        if (m == 1) access(o, ids[0]);
        else batch_access(o, ids);
    }
    /* Measurement phase -- measure stash */
    // Compute stats -- variance via Welford's method
    std::uint64_t n = 0;
    double mean_stash = 0.0, M2 = 0.0;
    // hist[s] = number of batches after which the stash held exactly s blocks
    std::vector<std::uint64_t> hist;
    for (std::uint64_t t = 0; t < new_T_meas; ++t) {
        std::vector<u32> ids(m);
        std::generate(ids.begin(), ids.end(), next_id);
        if (m == 1) access(o, ids[0]);
        else batch_access(o, ids);

        const std::size_t sz = o.stash.size();
        if (sz >= hist.size()) hist.resize(sz + 1, 0);
        ++hist[sz];

        const double s = sz;
        o.max_stash = std::max(o.max_stash, o.stash.size());
        ++n;
        const double delta = s - mean_stash;
        mean_stash += delta / n;
        M2 += delta * (s - mean_stash);
    }
    const double std_stash = std::sqrt(M2 / n);
    print_csv(csv, hcsv, L, k, m, hist, o.max_stash, mean_stash, std_stash, check_invariant(o));
}

int main() {
    fs::path dir_path = "./results/exp3";

    try {
        if (fs::create_directories(dir_path)) {
            std::cout << "Directory created successfully.\n";
        } else {
            std::cout << "Directory already exists or could not be created.\n";
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
    const fs::path csv_path = dir_path / "exp3.csv";
    const bool need_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) throw std::runtime_error("Could not open exp3.csv for writing");

    csv << std::setprecision(10);

    if (need_header)
        csv << "L,k,m,max_stash,mean_stash,std_stash,invariant\n";

    // Stash histogram, one row per (config, stash size). Rewritten on every
    // run (unlike exp3.csv, which is appended to).
    std::ofstream hcsv(dir_path / "exp3_hist.csv", std::ios::trunc);
    if (!hcsv) throw std::runtime_error("Could not open exp3_hist.csv for writing");
    hcsv << "L,k,m,stash,count\n";

    std::vector<u32> heights = {16,17,18,19,20};
    std::vector<u32> ms = {1,2,4,8,16,32,64};
    std::vector<u32> ks = {2}; // k=0 is uniform Z=4

    for (const auto& k : ks) {
        for (const auto& L : heights) {
            for (const auto& m : ms) {
                ORAM o; init(o, L, [L, k](u32 level) {
                    if (level == L-k) return 2u;
                    return level < L-k ? 4u : 0u;
                });
                std::cout << "init done, invariant " << (check_invariant(o) ? "OK" : "BROKEN") << ", stash " << o.stash.size() << "\n";
                run_experiment(o, csv, hcsv, L, k, m);
            }
        }
    }
    return 0;
}