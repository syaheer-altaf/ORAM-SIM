/* LP-ORAM simulator
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
** NOTE: the code is written in such a way that the user must define their own rule for
** bucket sizes with respect to the bucket's level on the tree.
*/

#include <iostream>
#include <cstdint>
#include <vector>
#include <map>
#include <random>
#include <functional>
#include <algorithm>
using u32 = std::uint32_t;

static constexpr u32 DUMMY = UINT32_MAX;   // dummy-block marker in a bucket
using ZFunc = std::function<u32(u32)>;     // level -> bucket size

/* Helpers */
static inline u32 bit_width(u32 v) { return v ? 32u - __builtin_clz(v) : 0u; }          // bit width of v (0 for v == 0)
static inline u32 node_level(u32 v) { return bit_width(v) - 1; }                        // level of a heap-indexed node (root = 0)
static inline u32 leaf_node(u32 L, u32 x) { return (1u << L) + x; }                     // heap index of the leaf bucket for leaf label x
static inline u32 path_node(u32 L, u32 x, u32 d) { return leaf_node(L, x) >> (L - d); } // node on path(x) at level d

// deepest level at which a block assigned to leaf x may live on path(y)
static inline u32 deepest_level(u32 L, u32 x, u32 y) { return L - bit_width(x ^ y); }

struct ORAM {
    u32 L;
    std::vector<u32> server_storage;               // server holds buckets with slot = block ID or DUMMY
    std::vector<std::size_t> level_offset;         // where level d starts in server_storage
    std::vector<u32> pos_map;                      // block ID -> leaf label
    std::vector<u32> stash;                        // block IDs

    ZFunc z;                                       // level -> bucket size
    std::mt19937_64 rng;
    std::uniform_int_distribution<u32> leaf_dist;

    // scratch for batch_access: in_U[v] != 0 iff bucket v is in the current union of paths
    std::vector<bool> in_U;

    // statistics
    std::vector<size_t>  stash_sizes;
    std::size_t max_stash = 0;
    std::uint64_t accesses = 0;         // single accesses, or batches for batch_access
    std::uint64_t buckets_touched = 0;  // sum of |U| over batches
};

static inline u32 random_leaf(ORAM& o) { return o.leaf_dist(o.rng); }

// view of bucket v: its Z(level) consecutive slots in the flat server_storage
// (supports range-for, [], size() and assign(z, DUMMY) like the old per-bucket vector)
struct Bucket {
    u32* p;
    u32 n;
    u32* begin() const { return p; }
    u32* end() const { return p + n; }
    u32 size() const { return n; }
    u32& operator[](u32 s) const { return p[s]; }
    void assign(u32 z, u32 val) const { std::fill(p, p + z, val); }
};

static inline Bucket get_bucket(const ORAM& o, u32 v) {
    const u32 d = node_level(v);
    const u32 z = o.z(d);
    u32* base = const_cast<u32*>(o.server_storage.data());  // const only for check_invariant, which just reads
    return { base + o.level_offset[d] + std::size_t(v - (1u << d)) * z, z };
}

u32 access(ORAM& o, u32 id);

/* ORAM initialize function */
void init(ORAM& o, u32 L, ZFunc z, std::uint64_t seed = 1) {
    o.L = L;
    o.z = std::move(z);
    o.rng.seed(seed);
    o.leaf_dist = std::uniform_int_distribution<u32>(0, (1u << L) - 1);
    o.stash.clear();
    o.pos_map.assign(1u << L, 0);

    // complete binary tree of 2^(L+1) - 1 buckets (index 0 unused), all dummy
    const u32 nodes = 1u << (L + 1);
    o.level_offset.assign(L + 2, 0);
    for (u32 d = 0; d <= L; ++d)
        o.level_offset[d + 1] = o.level_offset[d] + (std::size_t(1) << d) * o.z(d);
    o.server_storage.assign(o.level_offset[L + 1], DUMMY);
    o.in_U.assign(nodes, 0);

    const u32 N = 1u << L;
    for (u32 i = 0; i < N; ++i)
        o.pos_map[i] = random_leaf(o);

    // write every block once (Access(write, i, block_i))
    for (u32 i = 0; i < N; ++i)
        access(o, i);

    o.max_stash = 0;
    o.accesses = 0;
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
        const u32 z = o.z(d);
        bucket.assign(z, DUMMY);                       // pad with dummies
        const u32 take = std::min<u32>(z, carry.size());
        for (u32 s = 0; s < take; ++s) {
            bucket[s] = carry.back();
            carry.pop_back();
        }
    }
    o.stash = std::move(carry);                        // whatever could not be placed
}

u32 access(ORAM& o, u32 id) {
    const u32 x = o.pos_map[id]; 
    o.pos_map[id] = random_leaf(o);
    read_path(o, x);

    // no data to read/replace -- on the very first write (init) the block
    // is not in the tree yet, so it enters the stash here
    if (std::find(o.stash.begin(), o.stash.end(), id) == o.stash.end())
        o.stash.push_back(id);

    write_path(o, x);

    // o.stash_sizes.push_back(o.stash.size());
    // o.max_stash = std::max(o.max_stash, o.stash.size());
    ++o.accesses;
    return id;
}

/* Batched access over the union of m paths
** Canonical order trick: with heap indexing, sorting node ids ascending IS the canonical
** order (root first, level by level, ties by increasing id), because a node at level d
** has id in [2^d, 2^(d+1)). Reverse canonical order is simply descending id.
** U is closed under parent (it is a union of root-to-leaf paths), so a block that does not
** fit in bucket v can always be carried to v >> 1, which comes later in reverse order.
*/
std::vector<u32> batch_access(ORAM& o, const std::vector<u32>& ids) {
    const u32 L = o.L;
    const std::size_t m = ids.size();                  // caller supplies exactly m ops

    std::vector<u32> xs(m);
    for (std::size_t k = 0; k < m; ++k) {
        xs[k] = o.pos_map[ids[k]];
        o.pos_map[ids[k]] = random_leaf(o);
    }

    // U = union of path(x_k), marked bottom-up; stop early once a shared ancestor is hit
    std::vector<u32> U;
    U.reserve(m * (L + 1));
    for (u32 x : xs)
        for (u32 v = leaf_node(L, x); v >= 1 && !o.in_U[v]; v >>= 1) {
            o.in_U[v] = 1;
            U.push_back(v);
        }
    std::sort(U.begin(), U.end());                     // canonical order

    // read every bucket of U into the stash
    for (u32 v : U)
        for (u32& slot : get_bucket(o, v))
            if (slot != DUMMY) {
                o.stash.push_back(slot);
                slot = DUMMY;
            }

    // no data -- every requested block is now in the stash (checked by check_invariant)

    // each stash block enters at the deepest bucket of U on its (new) path;
    // blocks whose path leaves U only at the root still enter at the root.
    std::map<u32, std::vector<u32>> pending;           // bucket -> candidate blocks
    for (u32 id : o.stash) {
        u32 v = leaf_node(L, o.pos_map[id]);
        while (!o.in_U[v]) v >>= 1;                    // root is always in U
        pending[v].push_back(id);
    }
    o.stash.clear();

    for (auto it = U.rbegin(); it != U.rend(); ++it) { // reverse canonical order
        const u32 v = *it;
        auto bucket = get_bucket(o, v);
        const u32 z = o.z(node_level(v));
        bucket.assign(z, DUMMY);                       // pad with dummies

        auto p = pending.find(v);
        if (p == pending.end()) continue;
        auto& cand = p->second;
        const u32 take = std::min<u32>(z, cand.size());
        for (u32 s = 0; s < take; ++s) {
            bucket[s] = cand.back();
            cand.pop_back();
        }
        if (!cand.empty()) {                           // leftovers move up to the parent
            if (v == 1) o.stash = std::move(cand);
            else {
                auto& up = pending[v >> 1];
                up.insert(up.end(), cand.begin(), cand.end());
            }
        }
        pending.erase(p);
    }

    for (u32 v : U) o.in_U[v] = 0;                     // reset scratch

    // o.stash_sizes.push_back(o.stash.size());
    // o.max_stash = std::max(o.max_stash, o.stash.size());
    ++o.accesses;
    o.buckets_touched += U.size();
    return ids;
}

/* Debug: every block is in the stash or on the path to its assigned leaf, exactly once */
bool check_invariant(const ORAM& o) {
    const u32 N = 1u << o.L;
    std::vector<std::uint8_t> seen(N, 0);
    for (u32 v = 1; v < (1u << (o.L + 1)); ++v) {
        if (get_bucket(o, v).size() != o.z(node_level(v))) return false;
        for (u32 id : get_bucket(o, v)) {
            if (id == DUMMY) continue;
            if (++seen[id] > 1) return false;
            u32 leaf = o.pos_map.at(id);
            if (path_node(o.L, leaf, node_level(v)) != v) return false;
        }
    }
    for (u32 id : o.stash)
        if (++seen[id] > 1) return false;
    for (u32 i = 0; i < N; ++i)
        if (seen[i] != 1) return false;
    return true;
}