// Pure C port of craftax.cu -- CPU, OpenMP-parallel over envs.
// Same game logic and state layout as the CUDA version.
#include "craftax.h"
#include "worker_pool.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

// ============================================================
// Map accessors / small helpers
// ============================================================
static inline int8_t map_get(const EnvState* s, int r, int c) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s->map_packed[idx];
    return (c & 1) ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
}

static inline void map_set(EnvState* s, int r, int c, int8_t val) {
    int idx = r * MAP_PACKED_ROW + (c >> 1);
    uint8_t byte = s->map_packed[idx];
    if (c & 1) s->map_packed[idx] = (byte & 0x0F) | ((val & 0x0F) << 4);
    else       s->map_packed[idx] = (byte & 0xF0) | (val & 0x0F);
}

static inline bool in_bounds(int r, int c) {
    return (unsigned)r < MAP_SIZE && (unsigned)c < MAP_SIZE;
}

static inline bool is_solid(int8_t b) {
    return b == BLK_WATER || b == BLK_STONE || b == BLK_TREE ||
           b == BLK_COAL  || b == BLK_IRON  || b == BLK_DIAMOND ||
           b == BLK_TABLE || b == BLK_FURNACE ||
           b == BLK_PLANT || b == BLK_RIPE_PLANT;
}

static inline int l1_dist(int r1, int c1, int r2, int c2) {
    int dr = r1 - r2; if (dr < 0) dr = -dr;
    int dc = c1 - c2; if (dc < 0) dc = -dc;
    return dr + dc;
}

static inline int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline int max_i(int a, int b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }
static inline int sign_i(int v) { return (v > 0) - (v < 0); }

static const int DIR_DR[5] = {0, 0, 0, -1, 1};
static const int DIR_DC[5] = {0, -1, 1, 0, 0};

static inline float rand_f(EnvState* s) { return pcg32_uniform(&s->rng); }
static inline int   rand_int(EnvState* s, int n) { return pcg32_range(&s->rng, n); }

// O(1) mob bitmap query: bit c of mob_bits[r] = "mob at (r,c)"
static inline bool has_mob_at(const EnvState* s, int r, int c) {
    if ((unsigned)r >= MAP_SIZE || (unsigned)c >= MAP_SIZE) return false;
    return ((s->mob_bits[r] >> c) & 1ULL) != 0;
}

// Bitmap maintenance helpers (one bit per tile)
static inline void mb_set(uint64_t* bits, int r, int c)   { bits[r] |=  (1ULL << c); }
static inline void mb_clear(uint64_t* bits, int r, int c) { bits[r] &= ~(1ULL << c); }
static inline bool mb_get(const uint64_t* bits, int r, int c) { return (bits[r] >> c) & 1ULL; }

static bool is_near_block(const EnvState* s, int8_t blk_type) {
    int pr = s->player_r, pc = s->player_c;
    static const int dr8[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int dc8[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = pr + dr8[i], nc = pc + dc8[i];
        if (in_bounds(nr, nc) && map_get(s, nr, nc) == blk_type) return true;
    }
    return false;
}

static inline int get_damage(const EnvState* s) {
    if (s->inv[11] > 0) return 5;
    if (s->inv[10] > 0) return 3;
    if (s->inv[9]  > 0) return 2;
    return 1;
}

// ============================================================
// Perlin noise worldgen
// ============================================================
static inline float perlin_interp(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Perlin kept for reference / API compat (no longer used in generate_world).
// The hot loop below evaluates all 4 layers inline, sharing the floor/frac/interp
// work across them.
__attribute__((unused))
static float perlin_2d(float x, float y, const float* cos_a, const float* sin_a, int grid) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float u = perlin_interp(fx), v = perlin_interp(fy);
    int i00 = (((x0  ) % grid) + grid) % grid * grid + (((y0  ) % grid) + grid) % grid;
    int i10 = (((x0+1) % grid) + grid) % grid * grid + (((y0  ) % grid) + grid) % grid;
    int i01 = (((x0  ) % grid) + grid) % grid * grid + (((y0+1) % grid) + grid) % grid;
    int i11 = (((x0+1) % grid) + grid) % grid * grid + (((y0+1) % grid) + grid) % grid;
    float n00 = cos_a[i00]*fx       + sin_a[i00]*fy;
    float n10 = cos_a[i10]*(fx-1.f) + sin_a[i10]*fy;
    float n01 = cos_a[i01]*fx       + sin_a[i01]*(fy-1.f);
    float n11 = cos_a[i11]*(fx-1.f) + sin_a[i11]*(fy-1.f);
    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return (nx0 + v * (nx1 - nx0) + 1.0f) * 0.5f;
}

// Exposed for the pool (see craftax_pool.c). Prototype in pool impl.
void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id);
static void generate_world(EnvState* s, uint64_t seed, uint64_t env_id) {
    _craftax_generate_world(s, seed, env_id);
}

void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id) {
    pcg32_seed(&s->rng, seed, env_id);

    for (int i = 0; i < MAP_SIZE * MAP_PACKED_ROW; i++)
        s->map_packed[i] = (uint8_t)(BLK_GRASS | (BLK_GRASS << 4));

    enum { GRID = 10, GRID_PAD = GRID * GRID + 16 };
    // Precompute (cos,sin) of gradient angles. Padded by +16 floats so AVX-512
    // loads at max row (x0=8 => row1=90) don't read out of bounds.
    float cos_a[4][GRID_PAD];
    float sin_a[4][GRID_PAD];
    for (int layer = 0; layer < 4; layer++) {
        for (int i = 0; i < GRID * GRID; i++) {
            float a = rand_f(s) * 2.0f * 3.14159265f;
            cos_a[layer][i] = cosf(a);
            sin_a[layer][i] = sinf(a);
        }
        // Zero the pad region -- values never used but memory must be readable.
        for (int i = GRID * GRID; i < GRID_PAD; i++) {
            cos_a[layer][i] = 0.0f;
            sin_a[layer][i] = 0.0f;
        }
    }

    float scale = (float)MAP_SIZE / (float)(GRID - 1);
    float inv_scale = 1.0f / scale;
    int center = MAP_SIZE / 2;

    // AVX-512 Perlin: fill noise[4][MAP_SIZE][MAP_SIZE] in 16-column chunks.
    // Uses permutexvar instead of gather for the gradient lookup (tables are
    // only 10 entries per row, all lanes fit in one ZMM).
    // Stack-allocated so each caller thread has its own (generate_world is
    // called concurrently by producer threads).
    _Alignas(64) float noise[4][MAP_SIZE][MAP_SIZE];
    {
        const __m512 c_lane = _mm512_setr_ps(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 half = _mm512_set1_ps(0.5f);
        const __m512 c6 = _mm512_set1_ps(6.0f);
        const __m512 c15 = _mm512_set1_ps(15.0f);
        const __m512 c10 = _mm512_set1_ps(10.0f);
        const __m512 invs = _mm512_set1_ps(inv_scale);
        const __m512i i_one = _mm512_set1_epi32(1);

        for (int r = 0; r < MAP_SIZE; r++) {
            float nr = (float)r * inv_scale;
            int x0 = (int)nr;
            float fx = nr - x0;
            float fx1 = fx - 1.0f;
            float u = perlin_interp(fx);
            int row0 = x0 * GRID, row1 = row0 + GRID;
            __m512 fx_v = _mm512_set1_ps(fx);
            __m512 fx1_v = _mm512_set1_ps(fx1);
            __m512 u_v  = _mm512_set1_ps(u);

            for (int c_base = 0; c_base < MAP_SIZE; c_base += 16) {
                __m512 c_v = _mm512_add_ps(_mm512_set1_ps((float)c_base), c_lane);
                __m512 nc_v = _mm512_mul_ps(c_v, invs);
                __m512i y0_v = _mm512_cvttps_epi32(nc_v);
                __m512 y0_f = _mm512_cvtepi32_ps(y0_v);
                __m512 fy_v  = _mm512_sub_ps(nc_v, y0_f);
                __m512 fy1_v = _mm512_sub_ps(fy_v, one);
                // v = fy^3 * (fy*(fy*6 - 15) + 10)  (smoothstep for Perlin)
                __m512 t = _mm512_fmsub_ps(fy_v, c6, c15);
                t = _mm512_fmadd_ps(fy_v, t, c10);
                __m512 fy2 = _mm512_mul_ps(fy_v, fy_v);
                __m512 fy3 = _mm512_mul_ps(fy2, fy_v);
                __m512 v_v = _mm512_mul_ps(fy3, t);
                __m512i y1_v = _mm512_add_epi32(y0_v, i_one);

                for (int k = 0; k < 4; k++) {
                    __m512 cos_r0 = _mm512_loadu_ps(&cos_a[k][row0]);
                    __m512 cos_r1 = _mm512_loadu_ps(&cos_a[k][row1]);
                    __m512 sin_r0 = _mm512_loadu_ps(&sin_a[k][row0]);
                    __m512 sin_r1 = _mm512_loadu_ps(&sin_a[k][row1]);

                    __m512 c00 = _mm512_permutexvar_ps(y0_v, cos_r0);
                    __m512 c10v= _mm512_permutexvar_ps(y0_v, cos_r1);
                    __m512 c01 = _mm512_permutexvar_ps(y1_v, cos_r0);
                    __m512 c11 = _mm512_permutexvar_ps(y1_v, cos_r1);
                    __m512 s00 = _mm512_permutexvar_ps(y0_v, sin_r0);
                    __m512 s10 = _mm512_permutexvar_ps(y0_v, sin_r1);
                    __m512 s01 = _mm512_permutexvar_ps(y1_v, sin_r0);
                    __m512 s11 = _mm512_permutexvar_ps(y1_v, sin_r1);

                    __m512 n00 = _mm512_fmadd_ps(c00,  fx_v,  _mm512_mul_ps(s00, fy_v));
                    __m512 n10 = _mm512_fmadd_ps(c10v, fx1_v, _mm512_mul_ps(s10, fy_v));
                    __m512 n01 = _mm512_fmadd_ps(c01,  fx_v,  _mm512_mul_ps(s01, fy1_v));
                    __m512 n11 = _mm512_fmadd_ps(c11,  fx1_v, _mm512_mul_ps(s11, fy1_v));

                    __m512 nx0 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n10, n00), n00);
                    __m512 nx1 = _mm512_fmadd_ps(u_v, _mm512_sub_ps(n11, n01), n01);
                    __m512 n = _mm512_fmadd_ps(v_v, _mm512_sub_ps(nx1, nx0), nx0);
                    n = _mm512_mul_ps(_mm512_add_ps(n, one), half);

                    _mm512_storeu_ps(&noise[k][r][c_base], n);
                }
            }
        }
    }

    // Tile-logic sweep reads from precomputed noise[][][].
    for (int r = 0; r < MAP_SIZE; r++) {
        for (int c = 0; c < MAP_SIZE; c++) {
            float water_noise    = noise[0][r][c];
            float mountain_noise = noise[1][r][c];
            float tree_noise     = noise[2][r][c];
            float path_noise     = noise[3][r][c];

            float dist = sqrtf((float)((r-center)*(r-center) + (c-center)*(c-center)));
            float prox = 1.0f - min_f(dist / 20.0f, 1.0f);

            float water_val = water_noise - prox * 0.3f;
            float mountain_val = mountain_noise - prox * 0.3f;

            int8_t blk = BLK_GRASS;
            if (water_val > 0.7f) {
                blk = BLK_WATER;
            } else if (water_val > 0.6f && water_val <= 0.75f) {
                blk = BLK_SAND;
            } else if (mountain_val > 0.7f) {
                blk = BLK_STONE;
                if (path_noise > 0.8f) blk = BLK_PATH;
                if (mountain_val > 0.85f && water_noise > 0.4f) blk = BLK_PATH;
                if (mountain_val > 0.85f && tree_noise > 0.7f)  blk = BLK_LAVA;
            }

            if (blk == BLK_STONE) {
                float ore = rand_f(s);
                if (ore < 0.005f && mountain_val > 0.8f) blk = BLK_DIAMOND;
                else if (ore < 0.035f) blk = BLK_IRON;
                else if (ore < 0.075f) blk = BLK_COAL;
            }
            if (blk == BLK_GRASS && tree_noise > 0.5f && rand_f(s) > 0.8f)
                blk = BLK_TREE;

            map_set(s, r, c, blk);
        }
    }

    map_set(s, center, center, BLK_GRASS);

    bool has_diamond = false;
    for (int r = 0; r < MAP_SIZE && !has_diamond; r++)
        for (int c = 0; c < MAP_SIZE && !has_diamond; c++)
            if (map_get(s, r, c) == BLK_DIAMOND) has_diamond = true;
    if (!has_diamond) {
        for (int att = 0; att < 1000; att++) {
            int r = rand_int(s, MAP_SIZE), c = rand_int(s, MAP_SIZE);
            if (map_get(s, r, c) == BLK_STONE) { map_set(s, r, c, BLK_DIAMOND); break; }
        }
    }

    s->player_r = center; s->player_c = center;
    s->player_dir = 4;
    s->health = 9; s->food = 9; s->drink = 9; s->energy = 9;
    s->is_sleeping = false;
    s->recover = s->hunger = s->thirst = s->fatigue = 0;

    memset(s->mob_bits, 0, sizeof(s->mob_bits));
    memset(s->zombie_bits, 0, sizeof(s->zombie_bits));
    memset(s->cow_bits, 0, sizeof(s->cow_bits));
    memset(s->skel_bits, 0, sizeof(s->skel_bits));
    memset(s->arrow_bits, 0, sizeof(s->arrow_bits));

    memset(s->inv, 0, sizeof(s->inv));
    memset(s->zombie_mask, 0, sizeof(s->zombie_mask));
    memset(s->zombie_hp,   0, sizeof(s->zombie_hp));
    memset(s->zombie_cd,   0, sizeof(s->zombie_cd));
    memset(s->cow_mask, 0, sizeof(s->cow_mask));
    memset(s->cow_hp,   0, sizeof(s->cow_hp));
    memset(s->skel_mask, 0, sizeof(s->skel_mask));
    memset(s->skel_hp,   0, sizeof(s->skel_hp));
    memset(s->skel_cd,   0, sizeof(s->skel_cd));
    memset(s->arrow_mask, 0, sizeof(s->arrow_mask));
    memset(s->plant_mask, 0, sizeof(s->plant_mask));
    memset(s->plant_age,  0, sizeof(s->plant_age));
    memset(s->achievements, 0, sizeof(s->achievements));
    s->timestep = 0;
    s->light_level = 1.0f;
}

// ============================================================
// Step sub-actions
// ============================================================
static void do_crafting(EnvState* s, int action) {
    bool t = is_near_block(s, BLK_TABLE);
    bool f = is_near_block(s, BLK_FURNACE);

    if (action == ACT_MAKE_WOOD_PICK  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[6]++; s->achievements[ACH_MAKE_WOOD_PICK] = true; }
    if (action == ACT_MAKE_STONE_PICK && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[7]++; s->achievements[ACH_MAKE_STONE_PICK] = true; }
    if (action == ACT_MAKE_IRON_PICK  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[8]++; s->achievements[ACH_MAKE_IRON_PICK] = true;
    }
    if (action == ACT_MAKE_WOOD_SWORD  && t && s->inv[0] >= 1) { s->inv[0]--; s->inv[9]++;  s->achievements[ACH_MAKE_WOOD_SWORD] = true; }
    if (action == ACT_MAKE_STONE_SWORD && t && s->inv[0] >= 1 && s->inv[1] >= 1) { s->inv[0]--; s->inv[1]--; s->inv[10]++; s->achievements[ACH_MAKE_STONE_SWORD] = true; }
    if (action == ACT_MAKE_IRON_SWORD  && t && f && s->inv[0] >= 1 && s->inv[1] >= 1 && s->inv[3] >= 1 && s->inv[2] >= 1) {
        s->inv[0]--; s->inv[1]--; s->inv[3]--; s->inv[2]--; s->inv[11]++; s->achievements[ACH_MAKE_IRON_SWORD] = true;
    }
}

static void do_action(EnvState* s) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;

    int dmg = get_damage(s);
    bool attacked = false;

    for (int i = 0; i < MAX_ZOMBIES && !attacked; i++) {
        if (s->zombie_mask[i] && s->zombie_r[i] == tr && s->zombie_c[i] == tc) {
            s->zombie_hp[i] -= dmg;
            if (s->zombie_hp[i] <= 0) {
                s->zombie_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->zombie_bits, tr, tc);
                s->achievements[ACH_DEFEAT_ZOMBIE] = true;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_COWS && !attacked; i++) {
        if (s->cow_mask[i] && s->cow_r[i] == tr && s->cow_c[i] == tc) {
            s->cow_hp[i] -= dmg;
            if (s->cow_hp[i] <= 0) {
                s->cow_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->cow_bits, tr, tc);
                s->achievements[ACH_EAT_COW] = true;
                s->food = (int8_t)min_i(9, s->food + 6);
                s->hunger = 0;
            }
            attacked = true;
        }
    }
    for (int i = 0; i < MAX_SKELETONS && !attacked; i++) {
        if (s->skel_mask[i] && s->skel_r[i] == tr && s->skel_c[i] == tc) {
            s->skel_hp[i] -= dmg;
            if (s->skel_hp[i] <= 0) {
                s->skel_mask[i] = false;
                mb_clear(s->mob_bits, tr, tc);
                mb_clear(s->skel_bits, tr, tc);
                s->achievements[ACH_DEFEAT_SKELETON] = true;
            }
            attacked = true;
        }
    }
    if (attacked) return;

    int8_t blk = map_get(s, tr, tc);
    switch (blk) {
        case BLK_TREE:
            map_set(s, tr, tc, BLK_GRASS);
            s->inv[0] = (int8_t)min_i(9, s->inv[0] + 1);
            s->achievements[ACH_COLLECT_WOOD] = true;
            break;
        case BLK_STONE:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[1] = (int8_t)min_i(9, s->inv[1] + 1);
                s->achievements[ACH_COLLECT_STONE] = true;
            } break;
        case BLK_COAL:
            if (s->inv[6] > 0 || s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[2] = (int8_t)min_i(9, s->inv[2] + 1);
                s->achievements[ACH_COLLECT_COAL] = true;
            } break;
        case BLK_IRON:
            if (s->inv[7] > 0 || s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[3] = (int8_t)min_i(9, s->inv[3] + 1);
                s->achievements[ACH_COLLECT_IRON] = true;
            } break;
        case BLK_DIAMOND:
            if (s->inv[8] > 0) {
                map_set(s, tr, tc, BLK_PATH);
                s->inv[4] = (int8_t)min_i(9, s->inv[4] + 1);
                s->achievements[ACH_COLLECT_DIAMOND] = true;
            } break;
        case BLK_GRASS:
            if (rand_f(s) < 0.1f) {
                s->inv[5] = (int8_t)min_i(9, s->inv[5] + 1);
                s->achievements[ACH_COLLECT_SAPLING] = true;
            } break;
        case BLK_WATER:
            s->drink = (int8_t)min_i(9, s->drink + 1);
            s->thirst = 0;
            s->achievements[ACH_COLLECT_DRINK] = true;
            break;
        case BLK_RIPE_PLANT:
            map_set(s, tr, tc, BLK_PLANT);
            s->food = (int8_t)min_i(9, s->food + 4);
            s->hunger = 0;
            s->achievements[ACH_EAT_PLANT] = true;
            for (int i = 0; i < MAX_PLANTS; i++) {
                if (s->plant_mask[i] && s->plant_r[i] == tr && s->plant_c[i] == tc) {
                    s->plant_age[i] = 0; break;
                }
            }
            break;
    }
}

static void place_block(EnvState* s, int action) {
    int tr = s->player_r + DIR_DR[s->player_dir];
    int tc = s->player_c + DIR_DC[s->player_dir];
    if (!in_bounds(tr, tc)) return;
    if (has_mob_at(s, tr, tc)) return;

    int8_t blk = map_get(s, tr, tc);
    if (action == ACT_PLACE_TABLE && s->inv[0] >= 2 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_TABLE); s->inv[0] -= 2;
        s->achievements[ACH_PLACE_TABLE] = true;
    } else if (action == ACT_PLACE_FURNACE && s->inv[1] >= 1 && !is_solid(blk)) {
        map_set(s, tr, tc, BLK_FURNACE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_FURNACE] = true;
    } else if (action == ACT_PLACE_STONE && s->inv[1] >= 1 && (!is_solid(blk) || blk == BLK_WATER)) {
        map_set(s, tr, tc, BLK_STONE); s->inv[1] -= 1;
        s->achievements[ACH_PLACE_STONE] = true;
    } else if (action == ACT_PLACE_PLANT && s->inv[5] >= 1 && blk == BLK_GRASS) {
        map_set(s, tr, tc, BLK_PLANT); s->inv[5] -= 1;
        s->achievements[ACH_PLACE_PLANT] = true;
        for (int i = 0; i < MAX_PLANTS; i++) {
            if (!s->plant_mask[i]) {
                s->plant_r[i] = tr; s->plant_c[i] = tc;
                s->plant_age[i] = 0; s->plant_mask[i] = true; break;
            }
        }
    }
}

static void move_player(EnvState* s, int action) {
    if (action < 1 || action > 4) return;
    int nr = s->player_r + DIR_DR[action];
    int nc = s->player_c + DIR_DC[action];
    s->player_dir = (int8_t)action;
    if (!in_bounds(nr, nc)) return;
    if (is_solid(map_get(s, nr, nc))) return;
    if (has_mob_at(s, nr, nc)) return;
    s->player_r = (int16_t)nr; s->player_c = (int16_t)nc;
}

static bool can_move_mob(const EnvState* s, int r, int c) {
    if (!in_bounds(r, c)) return false;
    int8_t blk = map_get(s, r, c);
    if (is_solid(blk)) return false;
    if (blk == BLK_LAVA) return false;
    if (has_mob_at(s, r, c)) return false;
    if (r == s->player_r && c == s->player_c) return false;
    return true;
}

static void update_mobs(EnvState* s) {
    int pr = s->player_r, pc = s->player_c;

    for (int i = 0; i < MAX_ZOMBIES; i++) {
        if (!s->zombie_mask[i]) continue;
        int zr = s->zombie_r[i], zc = s->zombie_c[i];
        int dist = l1_dist(zr, zc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->zombie_mask[i] = false;
            mb_clear(s->mob_bits, zr, zc);
            mb_clear(s->zombie_bits, zr, zc);
            continue;
        }
        if (dist <= 1 && s->zombie_cd[i] <= 0) {
            int dmg = s->is_sleeping ? 7 : 2;
            s->health -= dmg;
            s->zombie_cd[i] = 5;
            s->is_sleeping = false;
        }
        s->zombie_cd[i] = (int8_t)max_i(0, s->zombie_cd[i] - 1);

        int dr = 0, dc = 0;
        if (dist < 10 && rand_f(s) < 0.75f) {
            int adr = abs(pr - zr), adc = abs(pc - zc);
            if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = sign_i(pr - zr);
            else                                                dc = sign_i(pc - zc);
        } else {
            int d = rand_int(s, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = zr + dr, nc = zc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, zr, zc); mb_clear(s->zombie_bits, zr, zc);
            s->zombie_r[i] = (int16_t)nr; s->zombie_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->zombie_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_COWS; i++) {
        if (!s->cow_mask[i]) continue;
        int cr = s->cow_r[i], cc = s->cow_c[i];
        int dist = l1_dist(cr, cc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->cow_mask[i] = false;
            mb_clear(s->mob_bits, cr, cc);
            mb_clear(s->cow_bits, cr, cc);
            continue;
        }
        int d = rand_int(s, 8);
        if (d < 4) {
            int dr = DIR_DR[d+1], dc2 = DIR_DC[d+1];
            int nr = cr + dr, nc = cc + dc2;
            if (can_move_mob(s, nr, nc)) {
                mb_clear(s->mob_bits, cr, cc); mb_clear(s->cow_bits, cr, cc);
                s->cow_r[i] = (int16_t)nr; s->cow_c[i] = (int16_t)nc;
                mb_set(s->mob_bits, nr, nc);   mb_set(s->cow_bits, nr, nc);
            }
        }
    }

    for (int i = 0; i < MAX_SKELETONS; i++) {
        if (!s->skel_mask[i]) continue;
        int sr = s->skel_r[i], sc = s->skel_c[i];
        int dist = l1_dist(sr, sc, pr, pc);
        if (dist >= MOB_DESPAWN_DIST) {
            s->skel_mask[i] = false;
            mb_clear(s->mob_bits, sr, sc);
            mb_clear(s->skel_bits, sr, sc);
            continue;
        }

        if (dist >= 4 && dist <= 5 && s->skel_cd[i] <= 0) {
            for (int a = 0; a < MAX_ARROWS; a++) {
                if (!s->arrow_mask[a]) {
                    s->arrow_mask[a] = true;
                    s->arrow_r[a] = (int16_t)sr; s->arrow_c[a] = (int16_t)sc;
                    mb_set(s->arrow_bits, sr, sc);
                    int adr = abs(pr - sr), adc = abs(pc - sc);
                    s->arrow_dr[a] = (int8_t)((adr > 0) ? sign_i(pr - sr) : 0);
                    s->arrow_dc[a] = (int8_t)((adc > 0) ? sign_i(pc - sc) : 0);
                    break;
                }
            }
            s->skel_cd[i] = 4;
        }
        s->skel_cd[i] = (int8_t)max_i(0, s->skel_cd[i] - 1);

        int dr = 0, dc = 0;
        bool random_move = rand_f(s) < 0.15f;
        if (!random_move) {
            if (dist >= 10) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = sign_i(pr - sr);
                else                                                dc = sign_i(pc - sc);
            } else if (dist <= 3) {
                int adr = abs(pr - sr), adc = abs(pc - sc);
                if (adr > adc || (adr == adc && rand_f(s) < 0.5f)) dr = -sign_i(pr - sr);
                else                                                dc = -sign_i(pc - sc);
            } else {
                random_move = true;
            }
        }
        if (random_move) {
            int d = rand_int(s, 4);
            dr = DIR_DR[d+1]; dc = DIR_DC[d+1];
        }
        int nr = sr + dr, nc = sc + dc;
        if (can_move_mob(s, nr, nc)) {
            mb_clear(s->mob_bits, sr, sc); mb_clear(s->skel_bits, sr, sc);
            s->skel_r[i] = (int16_t)nr; s->skel_c[i] = (int16_t)nc;
            mb_set(s->mob_bits, nr, nc);   mb_set(s->skel_bits, nr, nc);
        }
    }

    for (int i = 0; i < MAX_ARROWS; i++) {
        if (!s->arrow_mask[i]) continue;
        int ar = s->arrow_r[i], ac = s->arrow_c[i];
        int nr = ar + s->arrow_dr[i];
        int nc = ac + s->arrow_dc[i];
        if (!in_bounds(nr, nc)) { s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue; }
        int8_t blk = map_get(s, nr, nc);
        if (is_solid(blk) && blk != BLK_WATER) {
            if (blk == BLK_FURNACE || blk == BLK_TABLE) map_set(s, nr, nc, BLK_PATH);
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        if (nr == pr && nc == pc) {
            s->health -= 2; s->is_sleeping = false;
            s->arrow_mask[i] = false; mb_clear(s->arrow_bits, ar, ac); continue;
        }
        mb_clear(s->arrow_bits, ar, ac);
        s->arrow_r[i] = (int16_t)nr; s->arrow_c[i] = (int16_t)nc;
        mb_set(s->arrow_bits, nr, nc);
    }
}

static bool try_spawn(EnvState* s, int min_dist, int max_dist,
                     bool need_grass, bool need_path, int* out_r, int* out_c) {
    int pr = s->player_r, pc = s->player_c;
    for (int att = 0; att < 20; att++) {
        int r = rand_int(s, MAP_SIZE), c = rand_int(s, MAP_SIZE);
        int dist = l1_dist(r, c, pr, pc);
        if (dist < min_dist || dist >= max_dist) continue;
        if (has_mob_at(s, r, c)) continue;
        if (r == pr && c == pc) continue;
        int8_t blk = map_get(s, r, c);
        if (need_grass && blk != BLK_GRASS) continue;
        if (need_path && blk != BLK_PATH) continue;
        if (!need_grass && !need_path && blk != BLK_GRASS && blk != BLK_PATH) continue;
        *out_r = r; *out_c = c; return true;
    }
    return false;
}

static void spawn_mobs(EnvState* s) {
    int n_cows = 0, n_zombies = 0, n_skels = 0;
    for (int i = 0; i < MAX_COWS;      i++) n_cows    += s->cow_mask[i];
    for (int i = 0; i < MAX_ZOMBIES;   i++) n_zombies += s->zombie_mask[i];
    for (int i = 0; i < MAX_SKELETONS; i++) n_skels   += s->skel_mask[i];

    if (n_cows < MAX_COWS && rand_f(s) < 0.1f) {
        int r, c;
        if (try_spawn(s, 3, MOB_DESPAWN_DIST, true, false, &r, &c)) {
            for (int i = 0; i < MAX_COWS; i++) if (!s->cow_mask[i]) {
                s->cow_mask[i] = true; s->cow_r[i] = (int16_t)r; s->cow_c[i] = (int16_t)c; s->cow_hp[i] = 3;
                mb_set(s->mob_bits, r, c); mb_set(s->cow_bits, r, c);
                break;
            }
        }
    }
    float zombie_chance = 0.02f + 0.1f * (1.0f - s->light_level) * (1.0f - s->light_level);
    if (n_zombies < MAX_ZOMBIES && rand_f(s) < zombie_chance) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, false, &r, &c)) {
            for (int i = 0; i < MAX_ZOMBIES; i++) if (!s->zombie_mask[i]) {
                s->zombie_mask[i] = true; s->zombie_r[i] = (int16_t)r; s->zombie_c[i] = (int16_t)c;
                s->zombie_hp[i] = 5; s->zombie_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->zombie_bits, r, c);
                break;
            }
        }
    }
    if (n_skels < MAX_SKELETONS && rand_f(s) < 0.05f) {
        int r, c;
        if (try_spawn(s, 9, MOB_DESPAWN_DIST, false, true, &r, &c)) {
            for (int i = 0; i < MAX_SKELETONS; i++) if (!s->skel_mask[i]) {
                s->skel_mask[i] = true; s->skel_r[i] = (int16_t)r; s->skel_c[i] = (int16_t)c;
                s->skel_hp[i] = 3; s->skel_cd[i] = 0;
                mb_set(s->mob_bits, r, c); mb_set(s->skel_bits, r, c);
                break;
            }
        }
    }
}

static void update_plants(EnvState* s) {
    for (int i = 0; i < MAX_PLANTS; i++) {
        if (!s->plant_mask[i]) continue;
        s->plant_age[i]++;
        if (s->plant_age[i] >= 600) {
            int r = s->plant_r[i], c = s->plant_c[i];
            if (in_bounds(r, c) && map_get(s, r, c) == BLK_PLANT)
                map_set(s, r, c, BLK_RIPE_PLANT);
        }
    }
}

static void update_intrinsics(EnvState* s, int action) {
    if (action == ACT_SLEEP && s->energy < 9) s->is_sleeping = true;
    if (s->energy >= 9 && s->is_sleeping) {
        s->is_sleeping = false;
        s->achievements[ACH_WAKE_UP] = true;
    }
    float sleep_mul = s->is_sleeping ? 0.5f : 1.0f;

    s->hunger += sleep_mul;
    if (s->hunger > 25.0f) { s->food--; s->hunger = 0; }

    s->thirst += sleep_mul;
    if (s->thirst > 20.0f) { s->drink--; s->thirst = 0; }

    if (s->is_sleeping) s->fatigue -= 1.0f; else s->fatigue += 1.0f;
    if (s->fatigue > 30.0f)   { s->energy--; s->fatigue = 0; }
    if (s->fatigue < -10.0f)  { s->energy = (int8_t)min_i(s->energy + 1, 9); s->fatigue = 0; }

    bool all_needs = (s->food > 0) && (s->drink > 0) && (s->energy > 0 || s->is_sleeping);
    if (all_needs) s->recover += s->is_sleeping ? 2.0f : 1.0f;
    else           s->recover += s->is_sleeping ? -0.5f : -1.0f;
    if (s->recover > 25.0f)  { s->health = (int8_t)min_i(s->health + 1, 9); s->recover = 0; }
    if (s->recover < -15.0f) { s->health--; s->recover = 0; }
}

// ============================================================
// Observation
// ============================================================
void craftax_build_obs(const EnvState* s, float* obs) {
    int pr = s->player_r, pc = s->player_c;
    int idx = 0;
    for (int dr = -3; dr <= 3; dr++) {
        for (int dc = -4; dc <= 4; dc++) {
            int r = pr + dr, c = pc + dc;
            int8_t blk = in_bounds(r, c) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
            // SIMD-friendly: wide zero then single scalar set.
            // 17 floats = 68B; compiler emits one AVX-512 ZMM store + tail.
            float* dst = obs + idx;
            for (int b = 0; b < NUM_BLOCK_TYPES; b++) dst[b] = 0.0f;
            dst[blk] = 1.0f;
            idx += NUM_BLOCK_TYPES;

            float mz = 0, mc = 0, ms = 0, ma = 0;
            if (in_bounds(r, c)) {
                mz = (float)mb_get(s->zombie_bits, r, c);
                mc = (float)mb_get(s->cow_bits,    r, c);
                ms = (float)mb_get(s->skel_bits,   r, c);
                ma = (float)mb_get(s->arrow_bits,  r, c);
            }
            obs[idx++] = mz; obs[idx++] = mc; obs[idx++] = ms; obs[idx++] = ma;
        }
    }
    for (int i = 0; i < NUM_INVENTORY; i++) obs[idx++] = (float)s->inv[i] / 10.0f;
    obs[idx++] = (float)s->health / 10.0f;
    obs[idx++] = (float)s->food   / 10.0f;
    obs[idx++] = (float)s->drink  / 10.0f;
    obs[idx++] = (float)s->energy / 10.0f;
    for (int d = 1; d <= 4; d++) obs[idx++] = (s->player_dir == d) ? 1.0f : 0.0f;
    obs[idx++] = s->light_level;
    obs[idx++] = s->is_sleeping ? 1.0f : 0.0f;
}

void craftax_build_obs_compact(const EnvState* s, uint8_t* obs) {
    int pr = s->player_r, pc = s->player_c;
    uint8_t mobs[OBS_MAP_ROWS * OBS_MAP_COLS];
    int tile = 0;
    for (int dr = -3; dr <= 3; dr++) {
        int r = pr + dr;
        bool row_ok = (unsigned)r < MAP_SIZE;
        // Extract 9 consecutive columns from each per-type row bitmap in one shot.
        uint64_t zb = row_ok ? s->zombie_bits[r] : 0;
        uint64_t cb = row_ok ? s->cow_bits[r]    : 0;
        uint64_t sb = row_ok ? s->skel_bits[r]   : 0;
        uint64_t ab = row_ok ? s->arrow_bits[r]  : 0;
        for (int dc = -4; dc <= 4; dc++) {
            int c = pc + dc;
            int8_t blk = (row_ok && (unsigned)c < MAP_SIZE) ? map_get(s, r, c) : BLK_OUT_OF_BOUNDS;
            obs[tile] = (uint8_t)blk;

            uint8_t m = 0;
            if (row_ok && (unsigned)c < MAP_SIZE) {
                uint64_t bit = 1ULL << c;
                m |= (zb & bit) ? 1 : 0;
                m |= (cb & bit) ? 2 : 0;
                m |= (sb & bit) ? 4 : 0;
                m |= (ab & bit) ? 8 : 0;
            }
            mobs[tile] = m;
            tile++;
        }
    }
    memcpy(obs + 63, mobs, 63);

    uint8_t* out = obs + 126;
    for (int i = 0; i < NUM_INVENTORY; i++) *out++ = (uint8_t)s->inv[i];
    *out++ = (uint8_t)s->health;
    *out++ = (uint8_t)s->food;
    *out++ = (uint8_t)s->drink;
    *out++ = (uint8_t)s->energy;
    *out++ = (uint8_t)s->player_dir;
    *out++ = (uint8_t)(s->is_sleeping ? 1 : 0);
    // light_level in [0,1] -> [0,255]
    float ll = s->light_level;
    if (ll < 0) ll = 0; else if (ll > 1.0f) ll = 1.0f;
    *out++ = (uint8_t)(ll * 255.0f + 0.5f);
}

// ============================================================
// Public single-env API
// ============================================================
void craftax_reset(EnvState* s, uint64_t seed, uint64_t env_id) {
    generate_world(s, seed, env_id);
}

void craftax_step(EnvState* s, int action, float* reward, int* done) {
    int old_health = s->health;
    bool old_ach[NUM_ACHIEVEMENTS];
    memcpy(old_ach, s->achievements, sizeof(old_ach));

    int eff_action = s->is_sleeping ? ACT_NOOP : action;

    do_crafting(s, eff_action);
    if (eff_action == ACT_DO) do_action(s);
    if (eff_action >= ACT_PLACE_STONE && eff_action <= ACT_PLACE_PLANT) place_block(s, eff_action);
    move_player(s, eff_action);
    update_mobs(s);
    spawn_mobs(s);
    update_plants(s);
    update_intrinsics(s, action);

    for (int i = 0; i < NUM_INVENTORY; i++) s->inv[i] = (int8_t)clamp_i(s->inv[i], 0, 9);

    s->timestep++;
    float t_frac = fmodf((float)s->timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cv = cosf(3.14159265f * t_frac);
    s->light_level = 1.0f - fabsf(cv * cv * cv);

    float ach_r = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++)
        ach_r += (float)(s->achievements[i] && !old_ach[i]);
    float hp_r = (float)(s->health - old_health) * 0.1f;
    *reward = ach_r + hp_r;

    bool d = (s->timestep >= MAX_TIMESTEPS) || (s->health <= 0);
    if (in_bounds(s->player_r, s->player_c) && map_get(s, s->player_r, s->player_c) == BLK_LAVA) d = true;
    *done = d ? 1 : 0;
}

// ============================================================
// Batched API (OpenMP)
// ============================================================
void craftax_reset_batch(EnvState* states, float* obs, int num_envs, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        craftax_reset(&states[i], seed, (uint64_t)i);
        craftax_build_obs(&states[i], obs + (size_t)i * OBS_DIM);
    }
}

void craftax_step_batch(EnvState* states, const int32_t* actions,
                        float* obs, float* rewards, int8_t* dones,
                        int num_envs, uint64_t reset_seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) craftax_reset(&states[i], reset_seed, (uint64_t)i + (uint64_t)num_envs);
        craftax_build_obs(&states[i], obs + (size_t)i * OBS_DIM);
    }
}

// ============================================================
// Thread-pool variant: no OpenMP, uses custom spin barrier.
// ============================================================
typedef struct {
    EnvState* states;
    const int32_t* actions;
    uint8_t* obs_compact;
    float* obs_float;
    float* rewards;
    int8_t* dones;
    int num_envs;
    uint64_t reset_seed;
} StepWork;

static void step_compact_worker(void* arg, int id, int nw, int begin, int end) {
    (void)id; (void)nw;
    StepWork* w = (StepWork*)arg;
    for (int i = begin; i < end; i++) {
        float r; int d;
        craftax_step(&w->states[i], (int)w->actions[i], &r, &d);
        w->rewards[i] = r;
        w->dones[i] = (int8_t)d;
        if (d) craftax_reset(&w->states[i], w->reset_seed,
                             (uint64_t)i + (uint64_t)w->num_envs);
        craftax_build_obs_compact(&w->states[i],
            w->obs_compact + (size_t)i * OBS_DIM_COMPACT);
    }
}

void craftax_step_batch_compact_tp(struct ThreadPool* tp,
                                   EnvState* states, const int32_t* actions,
                                   uint8_t* obs, float* rewards, int8_t* dones,
                                   int num_envs, uint64_t reset_seed) {
    StepWork w = {
        .states = states, .actions = actions,
        .obs_compact = obs, .obs_float = NULL,
        .rewards = rewards, .dones = dones,
        .num_envs = num_envs, .reset_seed = reset_seed,
    };
    worker_pool_run(tp, num_envs, step_compact_worker, &w);
}

void craftax_reset_batch_compact(EnvState* states, uint8_t* obs, int num_envs, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        craftax_reset(&states[i], seed, (uint64_t)i);
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
}

void craftax_step_batch_compact(EnvState* states, const int32_t* actions,
                                uint8_t* obs, float* rewards, int8_t* dones,
                                int num_envs, uint64_t reset_seed) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) craftax_reset(&states[i], reset_seed, (uint64_t)i + (uint64_t)num_envs);
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
}
