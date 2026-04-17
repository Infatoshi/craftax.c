#pragma once
#include <stdint.h>
#include <stdbool.h>

// ============================================================
// Constants (mirror craftax.cuh)
// ============================================================
#define MAP_SIZE 64
#define MAX_ZOMBIES 3
#define MAX_COWS 3
#define MAX_SKELETONS 2
#define MAX_ARROWS 3
#define MAX_PLANTS 10
#define NUM_ACHIEVEMENTS 22
#define NUM_ACTIONS 17
#define NUM_BLOCK_TYPES 17
#define OBS_DIM 1345
#define OBS_MAP_ROWS 7
#define OBS_MAP_COLS 9
#define OBS_MAP_CHANNELS 21

// Compact obs layout (uint8_t per env):
//   [0..63)    block_ids    (63 tiles, values 0..16)
//   [63..126)  mob bitmask  (bit0=zombie, bit1=cow, bit2=skel, bit3=arrow)
//   [126..138) inventory    (12 slots, 0..9)
//   [138..142) health,food,drink,energy (0..9)
//   [142]      player_dir   (0..4)
//   [143]      is_sleeping  (0/1)
//   [144]      light_level  (quantized 0..255)
#define OBS_DIM_COMPACT 145
#define NUM_INVENTORY 12
#define MAX_TIMESTEPS 10000
#define DAY_LENGTH 300
#define MOB_DESPAWN_DIST 14

#define MAP_PACKED_ROW 32
#define MAP_PACKED_SIZE (MAP_SIZE * MAP_PACKED_ROW)

#define BLK_INVALID       0
#define BLK_OUT_OF_BOUNDS 1
#define BLK_GRASS         2
#define BLK_WATER         3
#define BLK_STONE         4
#define BLK_TREE          5
#define BLK_WOOD          6
#define BLK_PATH          7
#define BLK_COAL          8
#define BLK_IRON          9
#define BLK_DIAMOND      10
#define BLK_TABLE        11
#define BLK_FURNACE      12
#define BLK_SAND         13
#define BLK_LAVA         14
#define BLK_PLANT        15
#define BLK_RIPE_PLANT   16

#define ACT_NOOP          0
#define ACT_LEFT          1
#define ACT_RIGHT         2
#define ACT_UP            3
#define ACT_DOWN          4
#define ACT_DO            5
#define ACT_SLEEP         6
#define ACT_PLACE_STONE   7
#define ACT_PLACE_TABLE   8
#define ACT_PLACE_FURNACE 9
#define ACT_PLACE_PLANT  10
#define ACT_MAKE_WOOD_PICK   11
#define ACT_MAKE_STONE_PICK  12
#define ACT_MAKE_IRON_PICK   13
#define ACT_MAKE_WOOD_SWORD  14
#define ACT_MAKE_STONE_SWORD 15
#define ACT_MAKE_IRON_SWORD  16

#define ACH_COLLECT_WOOD     0
#define ACH_PLACE_TABLE      1
#define ACH_EAT_COW          2
#define ACH_COLLECT_SAPLING  3
#define ACH_COLLECT_DRINK    4
#define ACH_MAKE_WOOD_PICK   5
#define ACH_MAKE_WOOD_SWORD  6
#define ACH_PLACE_PLANT      7
#define ACH_DEFEAT_ZOMBIE    8
#define ACH_COLLECT_STONE    9
#define ACH_PLACE_STONE     10
#define ACH_EAT_PLANT       11
#define ACH_DEFEAT_SKELETON 12
#define ACH_MAKE_STONE_PICK 13
#define ACH_MAKE_STONE_SWORD 14
#define ACH_WAKE_UP         15
#define ACH_PLACE_FURNACE   16
#define ACH_COLLECT_COAL    17
#define ACH_COLLECT_IRON    18
#define ACH_COLLECT_DIAMOND 19
#define ACH_MAKE_IRON_PICK  20
#define ACH_MAKE_IRON_SWORD 21

// ============================================================
// PCG32 RNG -- tiny, fast, per-env
// ============================================================
typedef struct { uint64_t state; uint64_t inc; } pcg32_t;

static inline uint32_t pcg32_next(pcg32_t* r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

static inline void pcg32_seed(pcg32_t* r, uint64_t seed, uint64_t seq) {
    r->state = 0;
    r->inc = (seq << 1u) | 1u;
    pcg32_next(r);
    r->state += seed;
    pcg32_next(r);
}

static inline float pcg32_uniform(pcg32_t* r) {
    // [0,1) with 24-bit mantissa
    return (pcg32_next(r) >> 8) * (1.0f / 16777216.0f);
}

static inline int pcg32_range(pcg32_t* r, int n) {
    // Bounded by n; unbiased enough for gameplay
    return (int)(pcg32_next(r) % (uint32_t)n);
}

// ============================================================
// Game State (per environment)
// ============================================================
typedef struct __attribute__((aligned(64))) {
    // Packed 4-bit map: nibble c%2 of byte [r*32 + c/2]
    uint8_t map_packed[MAP_SIZE * MAP_PACKED_ROW];

    // Per-row occupancy bitmaps (bit c = "mob/arrow at column c of row r")
    // mob_bits covers zombie|cow|skel (used by has_mob_at / can_move_mob).
    // The per-type bitmaps accelerate obs construction.
    uint64_t mob_bits[MAP_SIZE];
    uint64_t zombie_bits[MAP_SIZE];
    uint64_t cow_bits[MAP_SIZE];
    uint64_t skel_bits[MAP_SIZE];
    uint64_t arrow_bits[MAP_SIZE];

    int16_t player_r, player_c;
    int8_t player_dir;

    int8_t health, food, drink, energy;
    bool is_sleeping;
    float recover, hunger, thirst, fatigue;

    int8_t inv[NUM_INVENTORY];

    int16_t zombie_r[MAX_ZOMBIES], zombie_c[MAX_ZOMBIES];
    int8_t zombie_hp[MAX_ZOMBIES], zombie_cd[MAX_ZOMBIES];
    bool zombie_mask[MAX_ZOMBIES];

    int16_t cow_r[MAX_COWS], cow_c[MAX_COWS];
    int8_t cow_hp[MAX_COWS];
    bool cow_mask[MAX_COWS];

    int16_t skel_r[MAX_SKELETONS], skel_c[MAX_SKELETONS];
    int8_t skel_hp[MAX_SKELETONS], skel_cd[MAX_SKELETONS];
    bool skel_mask[MAX_SKELETONS];

    int16_t arrow_r[MAX_ARROWS], arrow_c[MAX_ARROWS];
    int8_t arrow_dr[MAX_ARROWS], arrow_dc[MAX_ARROWS];
    bool arrow_mask[MAX_ARROWS];

    int16_t plant_r[MAX_PLANTS], plant_c[MAX_PLANTS];
    int16_t plant_age[MAX_PLANTS];
    bool plant_mask[MAX_PLANTS];

    float light_level;
    bool achievements[NUM_ACHIEVEMENTS];
    int32_t timestep;

    pcg32_t rng;
} EnvState;

// ============================================================
// Public API
// ============================================================
void craftax_reset(EnvState* s, uint64_t seed, uint64_t env_id);
void craftax_step(EnvState* s, int action, float* reward, int* done);
void craftax_build_obs(const EnvState* s, float* obs);
void craftax_build_obs_compact(const EnvState* s, uint8_t* obs);

// Batched helpers (OpenMP parallel over envs)
void craftax_reset_batch(EnvState* states, float* obs, int num_envs, uint64_t seed);
void craftax_step_batch(EnvState* states, const int32_t* actions,
                        float* obs, float* rewards, int8_t* dones,
                        int num_envs, uint64_t reset_seed);

void craftax_reset_batch_compact(EnvState* states, uint8_t* obs, int num_envs, uint64_t seed);
void craftax_step_batch_compact(EnvState* states, const int32_t* actions,
                                uint8_t* obs, float* rewards, int8_t* dones,
                                int num_envs, uint64_t reset_seed);

// ============================================================
// World-generation pool (pipelined resets)
// Producer threads pre-generate worlds; consumers pop on reset.
// ============================================================
typedef struct WorldPool WorldPool;

// capacity = ring size (slots). num_producers = background threads.
// Producer threads are pinned to SMT siblings of CCD1 cores (24..24+num_producers-1).
WorldPool* craftax_pool_create(int capacity, int num_producers, uint64_t master_seed);
void       craftax_pool_destroy(WorldPool* p);
void       craftax_pool_stats(WorldPool* p, uint64_t* produced, uint64_t* consumed,
                              uint64_t* fallbacks, int* ready_count);

// Same contract as craftax_step_batch_compact, but pops a pre-generated world
// on reset (falls back to inline generate_world if pool is empty).
void craftax_step_batch_compact_pool(WorldPool* pool,
                                     EnvState* states, const int32_t* actions,
                                     uint8_t* obs, float* rewards, int8_t* dones,
                                     int num_envs);

// ============================================================
// Worker-pool variants (use custom spin-barrier instead of OMP)
// ============================================================
struct ThreadPool;  // opaque; see worker_pool.h

void craftax_step_batch_compact_tp(struct ThreadPool* tp,
                                   EnvState* states, const int32_t* actions,
                                   uint8_t* obs, float* rewards, int8_t* dones,
                                   int num_envs, uint64_t reset_seed);

void craftax_step_batch_compact_pool_tp(struct ThreadPool* tp, WorldPool* pool,
                                        EnvState* states, const int32_t* actions,
                                        uint8_t* obs, float* rewards, int8_t* dones,
                                        int num_envs);
