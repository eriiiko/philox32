// curand_oracle.cu -- philox32.h (host) vs NVIDIA cuRAND's curand_Philox4x32_10
// (device), an INDEPENDENT implementation of the same primitive, on 200 000
// pseudo-random (ctr, key) pairs plus 12 adversarial edge rows.
//
// Usage:  curand_oracle <kat file>       (the KAT file is unused; same CLI as the others)
// Output protocol (parsed by tests/test_curand_oracle.py):
//   ORACLE N=<n> edge=<e> mismatches=<m> edge_mismatches=<me>
//   EDGE c0 c1 c2 c3 k0 k1 o0 o1 o2 o3      (the cuRAND outputs for the edge rows, hex)
//   RESULT: PASS | RESULT: FAIL | RESULT: NO_DEVICE (exit 3)
//
// Origin: written as an adversarial-review oracle for philox32 0.1.0 and
// adopted as a test.  The fuzz inputs come from splitmix64 (nothing to do with
// Philox), so the oracle does not share any arithmetic with the code under test.
//
// License: MIT, Copyright (c) 2026 Erik Steen.
#include "philox32.h"

#include <curand_kernel.h>
#include <cstdio>
#include <vector>

__global__ void oracle_kernel(const unsigned int* ctr, const unsigned int* key, unsigned int* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint4 c = make_uint4(ctr[4 * i + 0], ctr[4 * i + 1], ctr[4 * i + 2], ctr[4 * i + 3]);
    const uint2 k = make_uint2(key[2 * i + 0], key[2 * i + 1]);
    const uint4 r = curand_Philox4x32_10(c, k);
    out[4 * i + 0] = r.x; out[4 * i + 1] = r.y; out[4 * i + 2] = r.z; out[4 * i + 3] = r.w;
}

// splitmix64 host PRNG for the fuzz inputs.
static unsigned long long g_state = 0x123456789abcdefULL;
static unsigned int next_word()
{
    g_state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = g_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (unsigned int)((z ^ (z >> 31)) >> 16);
}

static bool cuda_ok(cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int main()
{
    int ndev = 0;
    const cudaError_t st = cudaGetDeviceCount(&ndev);
    if (st == cudaErrorNoDevice || (st == cudaSuccess && ndev == 0)) {
        std::printf("RESULT: NO_DEVICE\n");
        return 3;
    }
    if (!cuda_ok(st, "cudaGetDeviceCount")) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    const int N = 200000;
    std::vector<unsigned int> hc(4 * N), hk(2 * N), ho(4 * N);
    for (int i = 0; i < 4 * N; ++i) hc[i] = next_word();
    for (int i = 0; i < 2 * N; ++i) hk[i] = next_word();

    // Adversarial edge rows: the input classes where a wrong hi/lo split, a
    // wrong permutation slot, or a wrong key-bump order would show.
    const unsigned int E[][6] = {
        {0, 0, 0, 0, 0, 0},
        {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu},
        {0, 1, 0, 0, 0, 0},                      // ctr[1] only
        {0, 0, 0, 1, 0, 0},                      // ctr[3] only
        {0, 0, 0, 0, 0, 1},                      // key[1] only
        {0, 0, 0, 0, 1, 0},                      // key[0] only
        {0, 0, 0, 0, 0x61C88647u, 0x449B5F4Bu},  // both key words wrap through 0 while bumping
        {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u},
        {0, 0xFFFFFFFFu, 0, 0xFFFFFFFFu, 0, 0},  // only ctr[1] / ctr[3] set
        {0xFFFFFFFFu, 0, 0xFFFFFFFFu, 0, 0, 0},  // only ctr[0] / ctr[2] set
        {0, 0, 0, 0, 0xFFFFFFFFu, 0},            // key[0] all ones
        {0, 0, 0, 0, 0, 0xFFFFFFFFu},            // key[1] all ones
    };
    const int NE = (int)(sizeof(E) / sizeof(E[0]));
    for (int j = 0; j < NE; ++j) {
        for (int t = 0; t < 4; ++t) hc[4 * j + t] = E[j][t];
        for (int t = 0; t < 2; ++t) hk[2 * j + t] = E[j][4 + t];
    }

    unsigned int *dc = 0, *dk = 0, *dout = 0;
    bool ok = cuda_ok(cudaMalloc(&dc, 4 * N * sizeof(unsigned int)), "cudaMalloc ctr")
           && cuda_ok(cudaMalloc(&dk, 2 * N * sizeof(unsigned int)), "cudaMalloc key")
           && cuda_ok(cudaMalloc(&dout, 4 * N * sizeof(unsigned int)), "cudaMalloc out")
           && cuda_ok(cudaMemcpy(dc, hc.data(), 4 * N * sizeof(unsigned int), cudaMemcpyHostToDevice), "H2D ctr")
           && cuda_ok(cudaMemcpy(dk, hk.data(), 2 * N * sizeof(unsigned int), cudaMemcpyHostToDevice), "H2D key");
    if (ok) {
        oracle_kernel<<<(N + 255) / 256, 256>>>(dc, dk, dout, N);
        ok = cuda_ok(cudaGetLastError(), "launch")
          && cuda_ok(cudaDeviceSynchronize(), "sync")
          && cuda_ok(cudaMemcpy(ho.data(), dout, 4 * N * sizeof(unsigned int), cudaMemcpyDeviceToHost), "D2H out");
    }
    cudaFree(dc); cudaFree(dk); cudaFree(dout);
    if (!ok) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    int bad = 0, bad_edge = 0;
    for (int i = 0; i < N; ++i) {
        uint32_t out[4];
        philox32_4x32_10(&hc[4 * i], &hk[2 * i], out);
        for (int j = 0; j < 4; ++j) {
            if (out[j] != ho[4 * i + j]) {
                ++bad;
                if (i < NE) ++bad_edge;
                if (bad <= 3)
                    std::printf("MISMATCH i=%d j=%d philox32=%08x curand=%08x\n", i, j, (unsigned)out[j], ho[4 * i + j]);
                break;
            }
        }
    }
    std::printf("ORACLE N=%d edge=%d mismatches=%d edge_mismatches=%d\n", N, NE, bad, bad_edge);
    for (int i = 0; i < NE; ++i)
        std::printf("EDGE %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    hc[4 * i], hc[4 * i + 1], hc[4 * i + 2], hc[4 * i + 3], hk[2 * i], hk[2 * i + 1],
                    ho[4 * i], ho[4 * i + 1], ho[4 * i + 2], ho[4 * i + 3]);
    std::printf("RESULT: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
