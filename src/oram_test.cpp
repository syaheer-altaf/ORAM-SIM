#include "path_oram.cpp"
#include <cstdint>
#include <iostream>

using namespace path_oram;

int main() {
    constexpr std::uint64_t T_meas = 1000000;
    constexpr u32 L = 15;
    constexpr u32 Z = 2;
    constexpr double kappa = 2.60;
    // run Path ORAM access (stash size is bad for Z=2)
    std::cout << "\nRunning Path ORAM access...\n";
    ORAM o;
    init(o, L, Z);
    std::cout << "init done, invariant "
              << (check_invariant(o) ? "OK" : "BROKEN")
              << ", L = " << L << ", Z = " << Z
              << ", stash size = " << o.stash.size() << '\n';
    u32 idx = 0;
    for (std::uint64_t t = 0; t < T_meas; ++t)
        access(o, idx++ % (1u << L));
    std::cout << "\t- bandwidth = " << o.buckets_touched * Z << " blocks\n";
    std::cout << "\t- avg bandwidth = " << (o.buckets_touched * Z) / T_meas << " blocks\n";
    std::cout << "\t- invariant: " << (check_invariant(o) ? "OK" : "BROKEN") << '\n';
    std::cout << "\t- final stash size: " << o.stash.size() << " <--- this is bad\n";

    // run Path ORAM with modified BatchAccess
    std::cout << "\nRunning with modified BatchAccess...\n";
    o = ORAM{};  // release old memory and reset the struct
    init(o, L, Z);
    std::cout << "init done, invariant "
              << (check_invariant(o) ? "OK" : "BROKEN")
              << ", L = " << L << ", Z = " << Z
              << ", stash size = " << o.stash.size() << '\n';
    for (std::uint64_t t = 0; t < T_meas; ++t)
        mod_batch_access(o, {idx++ % (1u << L)}, kappa);
    std::cout << "\t- bandwidth = " << o.buckets_touched * Z << " blocks\n";
    std::cout << "\t- avg bandwidth = " << (o.buckets_touched * Z) / T_meas << " blocks\n";
    std::cout << "\t- invariant: " << (check_invariant(o) ? "OK" : "BROKEN") << '\n';
    std::cout << "\t- final stash size: " << o.stash.size() << " <--- this should be okay\n";

    return 0;
}