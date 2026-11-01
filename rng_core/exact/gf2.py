"""rng_core.exact.gf2 - GF(2) machinery: Berlekamp-Massey, bitset Gaussian elimination, symbolic linear generators.

Python ints are used as bitsets (XOR on big ints is fast and has no dependencies). Part 1 (BM) is also used by diag.py.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


# ── Berlekamp-Massey over GF(2) ──────────────────────────────────────────────────────────────────
def berlekamp_massey(bits: Iterable[int]) -> Tuple[int, int]:
    """Shortest LFSR for the bit sequence. Returns (L, C): C is the connection polynomial as an int bitset
    (bit i = c_i, c_0 = 1) with s_N = XOR_{i=1..L} c_i s_{N-i}.  O(n^2 / 64) via big-int XOR / popcount."""
    C, B, L, m, R = 1, 1, 0, 1, 0
    for N, s in enumerate(bits):
        R = (R << 1) | int(s)                         # bit k of R == s_{N-k}
        if (C & R).bit_count() & 1:
            T = C
            C ^= B << m
            if 2 * L <= N:
                L = N + 1 - L
                B = T
                m = 1
            else:
                m += 1
        else:
            m += 1
    return L, C


def bm_predict(bits: Sequence[int], L: int, C: int, steps: int) -> List[int]:
    """Roll the recurrence forward `steps` bits from the end of `bits` (uses only the last L bits)."""
    R = 0
    for s in bits[-max(L, 1):]:
        R = (R << 1) | int(s)
    out = []
    for _ in range(steps):
        nb = ((C & (R << 1)).bit_count()) & 1         # bit0 of R<<1 is 0, so c_0 does not contribute
        out.append(nb)
        R = (R << 1) | nb
    return out


def lc_pvalue(L: int, n: int) -> float:
    """Exact two-sided p-value of linear complexity L for a uniform random binary sequence of length n.
    #sequences with LC l: 1 (l=0), 2^(2l-1) (1<=l<=n/2), 2^(2(n-l)) (n/2<l<=n)."""
    import math
    if n < 2:
        return 1.0
    half = n // 2
    if L <= half:
        lg = (2 * L + 1 - n) - math.log2(3.0)                         # lower tail  ~ (2^(2L+1)+1) / (3 * 2^n)
        tail = 2.0 ** lg if lg > -1000 else 1e-300
        tail += 1.0 / (3.0 * 2.0 ** n) if n < 1000 else 0.0
    else:
        lg = (n - 2 * L + 2) - math.log2(3.0)                         # upper tail  ~ 4^(n-L+1) / (3 * 2^n)
        tail = 2.0 ** lg if lg > -1000 else 1e-300
    return float(min(1.0, max(2.0 * tail, 1e-300)))


# ── GF(2) linear systems ─────────────────────────────────────────────────────────────────────────
class GF2System:
    """Incremental row-reduced system over GF(2) with up to `nvars` unknowns.

    A row is (mask, rhs): mask is an int bitset over the unknowns, rhs the bit it must equal. add() reduces the row against
    the existing pivots; returns 'new' (rank grew), 'dep' (consistent, redundant) or 'bad' (inconsistent -> hypothesis false).
    """

    def __init__(self, nvars: int):
        self.nvars = nvars
        self.piv: dict = {}                    # leading bit -> (mask, rhs)
        self.inconsistent = False

    @property
    def rank(self) -> int:
        return len(self.piv)

    def add(self, mask: int, rhs: int) -> str:
        piv = self.piv
        while mask:
            lb = mask.bit_length() - 1
            p = piv.get(lb)
            if p is None:
                piv[lb] = (mask, rhs & 1)
                return "new"
            mask ^= p[0]
            rhs ^= p[1]
        if rhs & 1:
            self.inconsistent = True
            return "bad"
        return "dep"

    def solve(self) -> Optional[int]:
        """Solution bitset with free variables set to 0 (back substitution); None if inconsistent."""
        if self.inconsistent:
            return None
        sol = 0
        for lb in sorted(self.piv):                          # ascending: a pivot only involves bits <= lb
            mask, rhs = self.piv[lb]
            rest = mask & ~(1 << lb)
            v = rhs ^ ((rest & sol).bit_count() & 1)
            if v:
                sol |= 1 << lb
        return sol

    def determined(self) -> bool:
        return not self.inconsistent and self.rank == self.nvars


# ── symbolic linear generators ───────────────────────────────────────────────────────────────────
# A symbolic 32-bit word is a list of 32 ints: word[b] = bitset (over the unknown initial state bits) of bit b.
Word = List[int]


def sym_const(v: int) -> Word:
    """Constant word; constants are tracked in a separate mask so the system stays homogeneous (we only need linear
    parts: every generator below is purely linear, so no constants appear)."""
    return [0] * 32


def sym_shl(w: Word, k: int) -> Word:
    return [0] * k + w[: 32 - k]


def sym_shr(w: Word, k: int) -> Word:
    return w[k:] + [0] * k


def sym_xor(a: Word, b: Word) -> Word:
    return [x ^ y for x, y in zip(a, b)]


def sym_state_words(n_words: int) -> List[Word]:
    """n_words 32-bit words whose bits are independent unknowns 0..32*n_words-1."""
    return [[1 << (32 * i + b) for b in range(32)] for i in range(n_words)]


class SymXorshift32:
    nvars = 32

    def __init__(self):
        self.x = sym_state_words(1)[0]

    def next(self) -> Word:
        x = self.x
        x = sym_xor(x, sym_shl(x, 13))
        x = sym_xor(x, sym_shr(x, 17))
        x = sym_xor(x, sym_shl(x, 5))
        self.x = x
        return x


class SymXorshift128:
    nvars = 128

    def __init__(self):
        self.s = sym_state_words(4)

    def next(self) -> Word:
        x, y, z, w = self.s
        t = sym_xor(x, sym_shl(x, 11))
        nw = sym_xor(sym_xor(w, sym_shr(w, 19)), sym_xor(t, sym_shr(t, 8)))
        self.s = [y, z, w, nw]
        return nw


class SymLFSR32:
    """Galois LFSR with taps 0xD0000001 (see gens.lfsr32_next); output = whole state word after one shift."""
    nvars = 32

    def __init__(self):
        self.x = sym_state_words(1)[0]

    def next(self) -> Word:
        x = self.x
        lsb = x[0]
        nx = sym_shr(x, 1)
        poly = 0xD0000001
        nx = [nx[b] ^ (lsb if (poly >> b) & 1 else 0) for b in range(32)]
        self.x = nx
        return nx


class SymMT19937:
    """Symbolic MT19937. Unknowns are the 624 state words (19968 bits; the low 31 bits of word 0 never influence an
    output, so rank <= 19937). Output k is the TEMPERED k-th word after the first twist (as Python's getrandbits(32))."""
    nvars = 624 * 32
    N, M = 624, 397
    UPPER, LOWER, MAG = 0x80000000, 0x7FFFFFFF, 0x9908B0DF

    def __init__(self, start_idx: int = 0):
        # start_idx=0 (default): the unknowns are the NEXT 624 untempered output words (valid at any alignment of a real MT);
        # start_idx=624: the unknowns are a freshly seeded state, so the first output is preceded by a twist.
        self.mt = sym_state_words(self.N)
        self.i = start_idx

    def _twist(self) -> None:
        mt, N, M = self.mt, self.N, self.M
        for k in range(N):
            a, b = mt[k], mt[(k + 1) % N]
            y = [b[j] if j < 31 else a[j] for j in range(32)]                       # (a & UPPER) | (b & LOWER)
            sh = y[1:] + [0]                                                         # y >> 1
            low = y[0]                                                               # y & 1 selects MAG
            new = mt[(k + M) % N]
            nw = [new[j] ^ sh[j] ^ (low if (self.MAG >> j) & 1 else 0) for j in range(32)]
            mt[k] = nw
        self.i = 0

    @staticmethod
    def temper(y: Word) -> Word:
        y = sym_xor(y, sym_shr(y, 11))
        sh = sym_shl(y, 7)
        y = sym_xor(y, [sh[j] if (0x9D2C5680 >> j) & 1 else 0 for j in range(32)])
        sh = sym_shl(y, 15)
        y = sym_xor(y, [sh[j] if (0xEFC60000 >> j) & 1 else 0 for j in range(32)])
        return sym_xor(y, sym_shr(y, 18))

    def next(self) -> Word:
        if self.i >= self.N:
            self._twist()
        y = self.mt[self.i]
        self.i += 1
        return self.temper(y)


SYM_GENS = {"xorshift32": SymXorshift32, "xorshift128": SymXorshift128, "lfsr32": SymLFSR32, "mt19937": SymMT19937}
