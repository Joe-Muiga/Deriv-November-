"""tools/lattice_lcg.py - OFFLINE lattice / LLL state recovery for truncated LCGs (brief 4c).  NOT IMPLEMENTED.

Planned: from n output digits of an LCG with unknown modulus-2^k state, build the standard truncated-LCG lattice and recover the
state with LLL (fpylll / sympy, offline only; heavy libraries are never imported by the live bot). This covers the cases the live
solvers do not: glibc rand() TYPE_3-adjacent LCGs with truncated output, 48-bit LCGs (java.util.Random, drand48) seen through a
digit map. It is deliberately a stub: it raises NotImplementedError so no report can claim coverage that does not exist.
"""


def recover_state(digits, modulus_bits: int = 48, multiplier: int = 0x5DEECE66D, increment: int = 0xB):
    raise NotImplementedError("lattice (LLL) recovery of truncated LCG state is not implemented (brief 4c); see docs/v2/REPORT.md")


if __name__ == "__main__":
    raise SystemExit("not implemented")
