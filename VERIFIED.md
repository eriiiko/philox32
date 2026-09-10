# Verified runs

The device leg of the suite cannot run in CI (GitHub runners have no GPU), so
the device claim carries provenance here instead: one dated line per machine
where `pytest -q` ran with the host, device and cuRAND-oracle legs actually
executing (no skips).  Add a line when you verify on new hardware.

| Date | GPU (arch) | CUDA | Host compiler | Python / numpy | Result |
|---|---|---|---|---|---|
| 2026-09-10 | NVIDIA GeForce RTX 3070 (sm_86) | 12.4.131 | MSVC 14.44 (cl 19.44.35223, VS 2022 BuildTools) | CPython 3.11.7 / numpy 2.4.6 | host KAT 3/3 PASS, device KAT 3/3 PASS, HOSTDEV 67/67, cuRAND oracle 200000 pairs + 12 edge rows: 0 mismatches |
