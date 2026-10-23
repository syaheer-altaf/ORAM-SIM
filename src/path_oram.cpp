/* Path-ORAM simulator
** For convenience, N = 2^L
** Every access here is read -- no data inside a block
**
** Tree layout: heap indexing. root = node 1, children of v are 2v and 2v+1
**   level(v)        = floor(log2 v)                    (root is level 0, leaves are level L)
**   leaf node of x  = 2^L + x                          (x in [0, 2^L))
**   ancestor of v at level d = v >> (level(v) - d)
** A block assigned to leaf x may sit in bucket u (level d) on path(y) iff
**   d <= deepest common level of x and y = L - bitwidth(x ^ y)
**
*/

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
using u32 = std::uint32_t;

static constexpr u32 DUMMY = UINT32_MAX;   // dummy-block marker in a bucket

/* Helpers */
static inline u32 bit_width(u32 v) { return v ? 32u - __builtin_clz(v) : 0u; }          // bit width of v (0 for v == 0)
static inline u32 node_level(u32 v) { return bit_width(v) - 1; }                        // level of a heap-indexed node (root = 0)
static inline u32 leaf_node(u32 L, u32 x) { return (1u << L) + x; }                     // heap index of the leaf bucket for leaf label x
static inline u32 path_node(u32 L, u32 x, u32 d) { return leaf_node(L, x) >> (L - d); } // node on path(x) at level d

// deepest level at which a block assigned to leaf x may live on path(y)
static inline u32 deepest_level(u32 L, u32 x, u32 y) { return L - bit_width(x ^ y); }
namespace path_oram {
    struct ORAM {
        u32 L = 0;
        std::vector<u32> server_storage;               // server holds buckets with slot = block ID or DUMMY
        std::vector<std::size_t> level_offset;         // where level d starts in server_storage
        std::vector<u32> pos_map;                      // block ID -> leaf label
        std::vector<u32> stash;                        // block IDs

        u32 Z = 0;                                     // bucket size
        bool initialized = false;
        std::mt19937_64 rng;
        std::uniform_int_distribution<u32> leaf_dist;

        // scratch for batch_access: in_U[v] != 0 iff bucket v is in the current union of paths
        std::vector<bool> in_U;

        // statistics
        std::vector<std::size_t> stash_sizes; // optional trace; not automatically populated
        std::size_t max_stash = 0;        // largest post-operation stash since init
        std::uint64_t accesses = 0;      // calls: accesses or batches
        std::uint64_t logical_reads = 0; // logical read requests, including repeats
        std::uint64_t buckets_touched = 0; // total distinct bucket reads (and writes)
    };

    static inline u32 random_leaf(ORAM& o) { return o.leaf_dist(o.rng); }

    // view of bucket v: its Z consecutive slots in the flat server_storage
    // (supports range-for, [], size() and assign(z, DUMMY) like the old per-bucket vector)
    template <typename T> struct BucketView {
        T* p;
        u32 n;
        T* begin() const { return p; }
        T* end() const { return p + n; }
        u32 size() const { return n; }
        T& operator[](u32 s) const { return p[s]; }
        // This method is only used on mutable (non-const T) bucket views.
        void assign(u32 count, u32 value) const { std::fill(p, p + count, value); }
    };

    static inline BucketView<u32> get_bucket(ORAM& o, u32 v) {
        const u32 d = node_level(v);
        return {o.server_storage.data() + o.level_offset[d]
                + std::size_t(v - (1u << d)) * o.Z, o.Z};
    }

    static inline BucketView<const u32> get_bucket(const ORAM& o, u32 v) {
        const u32 d = node_level(v);
        return {o.server_storage.data() + o.level_offset[d]
                + std::size_t(v - (1u << d)) * o.Z, o.Z};
    }

    u32 access(ORAM& o, u32 id);

    /* ORAM initialize function */
    void init(ORAM& o, u32 L, u32 Z, std::uint64_t seed = 1) {
        // The indexing scheme requires 2^(L+1) to fit in a u32.
        if (L > 30) throw std::invalid_argument("L must be at most 30");
        if (Z == 0) throw std::invalid_argument("Z must be positive");
        o.initialized = false;
        o.L = L;
        o.Z = Z;
        o.rng.seed(seed);
        o.leaf_dist = std::uniform_int_distribution<u32>(0, (1u << L) - 1);
        o.stash.clear();
        o.pos_map.assign(1u << L, 0);

        // complete binary tree of 2^(L+1) - 1 buckets (index 0 unused), all dummy
        const u32 nodes = 1u << (L + 1);
        o.level_offset.assign(L + 2, 0);
        for (u32 d = 0; d <= L; ++d)
            o.level_offset[d + 1] = o.level_offset[d] + (std::size_t(1) << d) * o.Z;
        o.server_storage.assign(o.level_offset[L + 1], DUMMY);
        o.in_U.assign(nodes, 0);

        const u32 N = 1u << L;
        for (u32 i = 0; i < N; ++i)
            o.pos_map[i] = random_leaf(o);

        // Algorithm 1 initialization: place each new block in the stash,
        // then perform the logical write. Only its ID is stored in this simulator.
        o.initialized = true;
        for (u32 i = 0; i < N; ++i) {
            o.stash.push_back(i);
            access(o, i);
        }

        // Do not include initialization operations in experiment statistics.
        o.max_stash = 0;
        o.accesses = 0;
        o.logical_reads = 0;
        o.buckets_touched = 0;
        o.stash_sizes.clear();
    }

    /* Access functions
    ** an access is read and write-back a path
    */

    // read path(x) into the stash
    static void read_path(ORAM& o, u32 x) {
        for (u32 d = 0; d <= o.L; ++d) {
            auto bucket = get_bucket(o, path_node(o.L, x, d));
            for (u32& slot : bucket) {
                if (slot != DUMMY) {
                    o.stash.push_back(slot);
                    slot = DUMMY;
                }
            }
        }
    }

    // greedy eviction from leaf to root
    static void write_path(ORAM& o, u32 x) {
        const u32 L = o.L;

        // bucket stash blocks by the deepest level they can reach on path(x)
        std::vector<std::vector<u32>> by_level(L + 1);
        for (u32 id : o.stash)
            by_level[deepest_level(L, o.pos_map[id], x)].push_back(id);

        // walking leaf -> root, every block eligible at level d is also eligible above it,
        // so leftover candidates simply carry upward
        std::vector<u32> carry;
        for (u32 d = L + 1; d-- > 0;) {
            carry.insert(carry.end(), by_level[d].begin(), by_level[d].end());

            auto bucket = get_bucket(o, path_node(L, x, d));
            const u32 Z = o.Z;
            bucket.assign(Z, DUMMY);                       // pad with dummies
            const u32 take = std::min<u32>(Z, carry.size());
            for (u32 s = 0; s < take; ++s) {
                bucket[s] = carry.back();
                carry.pop_back();
            }
        }
        o.stash = std::move(carry);                        // whatever could not be placed
    }

    u32 access(ORAM& o, u32 id) {
        if (!o.initialized) throw std::logic_error("ORAM is not initialized");
        if (id >= o.pos_map.size()) throw std::out_of_range("invalid block ID");
        const u32 x = o.pos_map[id];
        o.pos_map[id] = random_leaf(o);
        read_path(o, x);

        // Algorithm 1, logical lookup. Initialization already inserted new
        // blocks into the stash. Missing blocks during reads indicate an error.
        if (std::find(o.stash.begin(), o.stash.end(), id) == o.stash.end())
            throw std::logic_error("requested block is missing from the stash");

        write_path(o, x);

        // o.max_stash = std::max(o.max_stash, o.stash.size());
        ++o.accesses;
        ++o.logical_reads;
        o.buckets_touched += o.L + 1;
        return id;
    }

    // modified BatchAccess 
    // this primitive supports batching with additional random eviction-only paths
    // parameterization: rho_Z = ceil(kappa_Z * m)
    // hence, each batch uses:
    //   m       real request paths
    //   rho_Z   additional random eviction-only paths
    //   m + rho_Z sampled paths (with overlaps merged in U)
    //
    // Example experiment parameters (not guarantees or bounds):
    //   Z = 4: kappa_Z = 0.30
    //   Z = 3: kappa_Z = 0.93
    //   Z = 2: kappa_Z = 2.60
    // For every m, the number of extra sampled paths is ceil(kappa_Z * m).
    std::vector<u32> mod_batch_access(ORAM& o, const std::vector<u32>& ids, double kappa_Z) {
        if (!o.initialized) throw std::logic_error("ORAM is not initialized");
        if (ids.empty()) throw std::invalid_argument("batch must be nonempty");
        if (!std::isfinite(kappa_Z) || kappa_Z < 0.0)
            throw std::invalid_argument("kappa_Z must be finite and nonnegative");
        for (u32 id : ids)
            if (id >= o.pos_map.size()) throw std::out_of_range("invalid block ID");

        const u32 L = o.L;
        const std::size_t m = ids.size();

        const long double scaled =
            std::ceil(static_cast<long double>(kappa_Z) * static_cast<long double>(m));
        const std::size_t max_paths =
            std::numeric_limits<std::size_t>::max() / (L + 1);
        if (m > max_paths || !std::isfinite(scaled) ||
            scaled > static_cast<long double>(max_paths - m))
            throw std::overflow_error("batch path count is too large");
        const std::size_t rho_Z = static_cast<std::size_t>(scaled);

        // Algorithm 3, lines 1-2: sample the extra independent uniform leaf
        // labels BEFORE any position-map updates.
        std::vector<u32> extra_paths(rho_Z);
        for (u32& z : extra_paths) z = random_leaf(o);

        // Algorithm 3, lines 3-6: update the map once per logical request,
        // including each occurrence of a repeated ID, in request order.
        std::vector<u32> xs(m);

        for (std::size_t k = 0; k < m; ++k) {
            xs[k] = o.pos_map[ids[k]];
            o.pos_map[ids[k]] = random_leaf(o);
        }

        // ------------------------------------------------------------
        // U = union of:
        //   (1) m ordinary old request paths
        //   (2) rho_Z independent random eviction-only paths
        //
        // thus there are m + rho_Z eviction paths in total
        //
        // the rho_Z additional paths DO NOT touch the position map and
        // DO NOT generate assignment paths
        // ------------------------------------------------------------
        std::vector<u32> U;
        U.reserve(std::min<std::size_t>(o.in_U.size() - 1,
                                        (m + rho_Z) * (L + 1)));

        auto add_path_to_U = [&](u32 x) {
            for (u32 v = leaf_node(L, x);
                v >= 1 && !o.in_U[v];
                v >>= 1)
            {
                o.in_U[v] = 1;
                U.push_back(v);
            }
        };

        // m real request eviction paths
        for (u32 x : xs)
            add_path_to_U(x);

        // rho_Z additional random eviction-only paths (sampled above)
        for (u32 z : extra_paths)
            add_path_to_U(z);

        std::sort(U.begin(), U.end());

        // ------------------------------------------------------------
        // everything below is exactly ordinary BatchAccess
        // ------------------------------------------------------------

        // read U into stash
        for (u32 v : U) {
            for (u32& slot : get_bucket(o, v)) {
                if (slot != DUMMY) {
                    o.stash.push_back(slot);
                    slot = DUMMY;
                }
            }
        }

        // Algorithm 3, lines 9-14: all read-only lookups must succeed.
        // No block payloads exist, so the result of each read is its ID.
        for (u32 id : ids)
            if (std::find(o.stash.begin(), o.stash.end(), id) == o.stash.end())
                throw std::logic_error("batch requested a missing block");

        // Place blocks at their deepest eligible touched bucket and propagate
        // any overflow upward, implementing reverse-canonical greedy eviction.
        std::map<u32, std::vector<u32>> pending;
        for (u32 id : o.stash) {
            u32 v = leaf_node(L, o.pos_map[id]);

            while (!o.in_U[v])
                v >>= 1;

            pending[v].push_back(id);
        }
        o.stash.clear();

        // same greedy reverse-canonical write-back.
        for (auto it = U.rbegin(); it != U.rend(); ++it) {
            const u32 v = *it;

            auto bucket = get_bucket(o, v);
            const u32 z = o.Z;
            bucket.assign(z, DUMMY);

            auto p = pending.find(v);
            if (p == pending.end())
                continue;

            auto& cand = p->second;
            const u32 take =
                std::min<u32>(z, static_cast<u32>(cand.size()));

            for (u32 s = 0; s < take; ++s) {
                bucket[s] = cand.back();
                cand.pop_back();
            }

            if (!cand.empty()) {
                if (v == 1) {
                    o.stash = std::move(cand);
                } else {
                    auto& up = pending[v >> 1];
                    up.insert(up.end(), cand.begin(), cand.end());
                }
            }

            pending.erase(p);
        }

        for (u32 v : U)
            o.in_U[v] = 0;

        // o.max_stash = std::max(o.max_stash, o.stash.size());
        ++o.accesses;
        o.logical_reads += m;
        o.buckets_touched += U.size();

        return ids;
    }

    /* Debug: every block is in the stash or on the path to its assigned leaf, exactly once */
    bool check_invariant(const ORAM& o) {
        if (!o.initialized || o.L > 30 || o.Z == 0) return false;
        const u32 N = 1u << o.L;
        if (o.pos_map.size() != N || o.in_U.size() != (std::size_t(1) << (o.L + 1)) ||
            o.level_offset.size() != o.L + 2 ||
            o.server_storage.size() != o.level_offset[o.L + 1]) return false;
        for (bool marked : o.in_U)
            if (marked) return false;
        for (u32 leaf : o.pos_map)
            if (leaf >= N) return false;
        std::vector<std::uint8_t> seen(N, 0);
        for (u32 v = 1; v < (1u << (o.L + 1)); ++v) {
            if (get_bucket(o, v).size() != o.Z) return false;
            for (u32 id : get_bucket(o, v)) {
                if (id == DUMMY) continue;
                if (id >= N || ++seen[id] > 1) return false;
                u32 leaf = o.pos_map.at(id);
                if (path_node(o.L, leaf, node_level(v)) != v) return false;
            }
        }
        for (u32 id : o.stash)
            if (id >= N || ++seen[id] > 1) return false;
        for (u32 i = 0; i < N; ++i)
            if (seen[i] != 1) return false;
        return true;
    }
}