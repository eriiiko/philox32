// kat_check_device.cu -- the KAT + cross-language dump computed ON THE GPU.
//
// Usage:  kat_check_device <kat_philox4x32_10.txt>
// Same output protocol as kat_check.cpp (KAT n/m, 64 PAIR lines, RESULT: ...),
// plus  HOSTDEV <matching>/<total>  -- device output vs the host header in the
// same binary, over the KAT rows and the 64 pairs.  Exit code 3 with
// "RESULT: NO_DEVICE" when no CUDA device is usable (the pytest wrapper skips).
//
// License: MIT, Copyright (c) 2026 Erik Steen.
#include "philox32.h"
#include "kat_common.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// One thread per (ctr, key) pair: out[i] = philox(ctr[i], key[i]), plus the two
// Q16 mappings of every output word, so the mappings are exercised on device too.
__global__ void philox_kernel(const uint32_t* ctr, const uint32_t* key, uint32_t* out,
                              uint32_t* uni, uint32_t* ang, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    philox32_4x32_10(ctr + 4 * i, key + 2 * i, out + 4 * i);
    for (int j = 0; j < 4; ++j) {
        uni[4 * i + j] = philox32_uniform_q16(out[4 * i + j]);
        ang[4 * i + j] = philox32_angle_q16(out[4 * i + j]);
    }
}

static bool cuda_ok(cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: kat_check_device <kat file>\n");
        return 2;
    }
    std::vector<KatRow> rows;
    if (!load_kat(argv[1], rows) || rows.empty()) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("RESULT: NO_DEVICE\n");
        return 3;
    }

    // Inputs: the KAT rows first, then the 64 derived pairs.
    const unsigned nkat = (unsigned)rows.size();
    const unsigned n = nkat + XLANG_PAIRS;
    std::vector<uint32_t> h_ctr(4 * n), h_key(2 * n), h_out(4 * n), h_uni(4 * n), h_ang(4 * n);
    for (unsigned i = 0; i < nkat; ++i) {
        for (int j = 0; j < 4; ++j) h_ctr[4 * i + j] = rows[i].ctr[j];
        for (int j = 0; j < 2; ++j) h_key[2 * i + j] = rows[i].key[j];
    }
    for (uint32_t p = 0; p < XLANG_PAIRS; ++p) {
        const unsigned i = nkat + p;
        derive_pair(rows, p, &h_ctr[4 * i], &h_key[2 * i]);
    }

    uint32_t *d_ctr = 0, *d_key = 0, *d_out = 0, *d_uni = 0, *d_ang = 0;
    bool ok = cuda_ok(cudaMalloc(&d_ctr, 4 * n * sizeof(uint32_t)), "cudaMalloc ctr")
           && cuda_ok(cudaMalloc(&d_key, 2 * n * sizeof(uint32_t)), "cudaMalloc key")
           && cuda_ok(cudaMalloc(&d_out, 4 * n * sizeof(uint32_t)), "cudaMalloc out")
           && cuda_ok(cudaMalloc(&d_uni, 4 * n * sizeof(uint32_t)), "cudaMalloc uni")
           && cuda_ok(cudaMalloc(&d_ang, 4 * n * sizeof(uint32_t)), "cudaMalloc ang")
           && cuda_ok(cudaMemcpy(d_ctr, h_ctr.data(), 4 * n * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D ctr")
           && cuda_ok(cudaMemcpy(d_key, h_key.data(), 2 * n * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D key");
    if (ok) {
        philox_kernel<<<(n + 127) / 128, 128>>>(d_ctr, d_key, d_out, d_uni, d_ang, n);
        ok = cuda_ok(cudaGetLastError(), "launch")
          && cuda_ok(cudaDeviceSynchronize(), "sync")
          && cuda_ok(cudaMemcpy(h_out.data(), d_out, 4 * n * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H out")
          && cuda_ok(cudaMemcpy(h_uni.data(), d_uni, 4 * n * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H uni")
          && cuda_ok(cudaMemcpy(h_ang.data(), d_ang, 4 * n * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H ang");
    }
    cudaFree(d_ctr); cudaFree(d_key); cudaFree(d_out); cudaFree(d_uni); cudaFree(d_ang);
    if (!ok) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    // KAT: device output vs the published vectors.
    unsigned passed = 0;
    for (unsigned i = 0; i < nkat; ++i) {
        bool same = true;
        for (int j = 0; j < 4; ++j) same = same && (h_out[4 * i + j] == rows[i].out[j]);
        if (same) ++passed;
    }
    std::printf("KAT %u/%u\n", passed, nkat);

    // Host vs device over everything, block output AND both mappings.
    unsigned agree = 0;
    for (unsigned i = 0; i < n; ++i) {
        uint32_t ref[4];
        philox32_4x32_10(&h_ctr[4 * i], &h_key[2 * i], ref);
        bool same = true;
        for (int j = 0; j < 4; ++j) {
            same = same && (ref[j] == h_out[4 * i + j])
                        && (philox32_uniform_q16(ref[j]) == h_uni[4 * i + j])
                        && (philox32_angle_q16(ref[j]) == h_ang[4 * i + j]);
        }
        if (same) ++agree;
    }
    std::printf("HOSTDEV %u/%u\n", agree, n);

    // PAIR lines from the DEVICE results (print_pair recomputes the mappings on
    // the host from the device words; the HOSTDEV line above covers the device
    // mappings themselves).
    for (uint32_t p = 0; p < XLANG_PAIRS; ++p) {
        const unsigned i = nkat + p;
        print_pair(p, &h_ctr[4 * i], &h_key[2 * i], &h_out[4 * i]);
    }

    const bool pass = (passed == nkat) && (agree == n);
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
