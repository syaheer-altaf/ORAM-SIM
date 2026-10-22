#include "path_oram.cpp"

using namespace path_oram;
int main() {
    ORAM o;
    init(o, 15, 4);
    std::cout << "init done, invariant " << (check_invariant(o) ? "OK" : "BROKEN") << ", stash " << o.stash.size() << "\n";
}