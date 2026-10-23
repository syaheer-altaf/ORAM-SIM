#include "../src/path_oram.cpp"
#include <cstdint>
#include <iostream>

using namespace path_oram;

int main() {
    constexpr std::uint64_t T_warm = 1000000;
    constexpr std::uint64_t T_meas = 1000000;
    constexpr u32 L = 15;
    constexpr u32 Z2 = 2;
    constexpr u32 Z4 = 4;
    constexpr double kappa = 2.60;

    std::cout << "Number of accesses used:\n\t- T_warm = " << T_warm << "\n\t- T_meas = " << T_meas << "\n";

    // run Path ORAM access (stash size is bad for Z=2)
    std::cout << "\nRunning Path ORAM access with Z = 2 (bad stash)...\n";
    ORAM o;
    init(o, L, Z2);
    std::cout << "init done, invariant "
              << (check_invariant(o) ? "OK" : "BROKEN")
              << ", L = " << o.L << ", Z = " << o.Z
              << ", stash size = " << o.stash.size() << '\n';
    u32 idx = 0;
    // warm-up phase (no need to measure stash)
    for (std::uint64_t t = 0; t < T_warm; ++t) {
        access(o, idx++ % (1u << L));
    }
    // measure stash here
    for (std::uint64_t t = 0; t < T_meas; ++t) {
        access(o, idx++ % (1u << L));
        o.max_stash = std::max(o.max_stash, o.stash.size());
    }
    std::cout << "\nReport:\n";
    std::cout << "\t- avg bandwidth = " << (2 * o.buckets_touched * o.Z) / (T_warm + T_meas) << " blocks\n";
    std::cout << "\t- storage size = " << o.server_storage.size() << " blocks\n";
    std::cout << "\t- invariant: " << (check_invariant(o) ? "OK" : "BROKEN") << '\n';
    std::cout << "\t- max stash size: " << o.max_stash << " <--- this is bad\n";

    // run Path ORAM with modified BatchAccess
    std::cout << "\nRunning with modified BatchAccess with Z = 2...\n";
    o = ORAM{};  // release old memory and reset the struct
    init(o, L, Z2);
    std::cout << "init done, invariant "
              << (check_invariant(o) ? "OK" : "BROKEN")
              << ", L = " << o.L << ", Z = " << o.Z
              << ", stash size = " << o.stash.size() << '\n';
    // warm-up phase
    for (std::uint64_t t = 0; t < T_warm; ++t) {
        mod_batch_access(o, {idx++ % (1u << L)}, kappa);
    }
    // measurement phase
    for (std::uint64_t t = 0; t < T_meas; ++t) {
        mod_batch_access(o, {idx++ % (1u << L)}, kappa);
        o.max_stash = std::max(o.max_stash, o.stash.size());
    }
    std::cout << "\nReport:\n";
    std::cout << "\t- avg bandwidth = " << (2 * o.buckets_touched * o.Z) / (T_warm + T_meas) << " blocks\n";
    std::cout << "\t- storage size = " << o.server_storage.size() << " blocks\n";
    std::cout << "\t- invariant: " << (check_invariant(o) ? "OK" : "BROKEN") << '\n';
    std::cout << "\t- max stash size: " << o.max_stash << " <--- this should be okay\n";

    return 0;
}