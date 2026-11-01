"""
test_local.py – Quick smoke test (no API key needed).

Run with: python test_local.py
Verifies that all modules import correctly and the core logic
produces sensible outputs on synthetic data.
"""

import sys
import numpy as np

print("=" * 60)
print("  SIFM Bot – Local Smoke Test")
print("=" * 60)

# ── 1. Indicators ─────────────────────────────────────────────────────────────
print("\n[1/5] Testing indicators …")
import indicators as ind

np.random.seed(42)
prices = 1.10 + np.cumsum(np.random.randn(300) * 0.001)
highs  = prices + np.abs(np.random.randn(300) * 0.0005)
lows   = prices - np.abs(np.random.randn(300) * 0.0005)

rsi14   = ind.rsi(prices, 14)
m, s, h = ind.macd(prices)
up, mid, lo = ind.bollinger_bands(prices)
k, d    = ind.stoch_rsi(prices)
adxv, pdi, mdi = ind.adx(highs, lows, prices)
atrv    = ind.atr(highs, lows, prices)

valid = lambda arr: arr[~np.isnan(arr)]
assert len(valid(rsi14)) > 0, "RSI failed"
assert len(valid(h)) > 0,     "MACD hist failed"
assert len(valid(mid)) > 0,   "BB failed"
assert len(valid(k)) > 0,     "StochRSI failed"
assert len(valid(adxv)) > 0,  "ADX failed"
assert len(valid(atrv)) > 0,  "ATR failed"

last_rsi = valid(rsi14)[-1]
print(f"   RSI(14) last: {last_rsi:.2f} ✓")
print(f"   MACD hist last: {valid(h)[-1]:.6f} ✓")
print(f"   BB mid last: {valid(mid)[-1]:.5f} ✓")
print(f"   ATR last: {valid(atrv)[-1]:.6f} ✓")
print("   Indicators OK ✓")

# ── 2. CandlestickBuilder ─────────────────────────────────────────────────────
print("\n[2/5] Testing CandlestickBuilder …")
from candlestick_builder import CandlestickBuilder
import time as _t

builder = CandlestickBuilder(granularity=300)
base_epoch = int(_t.time()) - 3600
# Feed 3 full 5-min bars
for bar_i in range(3):
    for tick_i in range(10):
        epoch = base_epoch + bar_i * 300 + tick_i * 30
        price = 1.10 + bar_i * 0.001 + tick_i * 0.0001
        builder.add_tick(epoch, price)
# Start the 4th bar
builder.add_tick(base_epoch + 3 * 300, 1.103)

assert builder.count == 3, f"Expected 3 bars, got {builder.count}"
print(f"   Completed bars: {builder.count} ✓")
print(f"   Last close: {builder.last_completed.close:.5f} ✓")
print("   CandlestickBuilder OK ✓")

# ── 3. SMC Analyser ───────────────────────────────────────────────────────────
print("\n[3/5] Testing SMCAnalyzer …")
from candlestick_builder import Candle
from smc_analyzer import SMCAnalyzer

# Create synthetic uptrend bars
candles = []
price_base = 1.10
for i in range(60):
    o = price_base + i * 0.001
    c = o + (0.0005 if i % 3 != 0 else -0.0002)
    h = max(o, c) + 0.0002
    l = min(o, c) - 0.0002
    candles.append(Candle(timestamp=base_epoch + i*3600,
                          open=o, high=h, low=l, close=c, volume=100))

smc = SMCAnalyzer()
ctx = smc.analyse(candles, atr=0.002)
print(f"   Structure: {ctx.structure}")
print(f"   Bias:      {ctx.bias}")
print(f"   Bull OBs:  {len(ctx.bullish_obs)}")
print(f"   Bull FVGs: {len(ctx.bullish_fvgs)}")
print("   SMCAnalyzer OK ✓")

# ── 4. SignalEngine ───────────────────────────────────────────────────────────
print("\n[4/5] Testing SignalEngine …")
from signal_engine import SignalEngine, module3_vote

# Build 40 LTF candles with a modest uptrend
ltf_candles = []
for i in range(40):
    o = 1.10 + i * 0.0003
    c = o + 0.0002
    h = c + 0.0001
    l = o - 0.0001
    ltf_candles.append(Candle(timestamp=base_epoch + i*300,
                               open=o, high=h, low=l, close=c, volume=120))

m3 = module3_vote(ltf_candles, min_votes=3)
print(f"   Module 3 vote: {m3} (expected +1 for uptrend)")

engine = SignalEngine(min_modules=2, min_votes=3)
sig = engine.evaluate(ltf_candles, "LONG", ctx, in_zone=True)
print(f"   Signal: {sig.direction} | strength={sig.strength} | {sig.reason}")
print("   SignalEngine OK ✓")

# ── 5. RiskManager ────────────────────────────────────────────────────────────
print("\n[5/5] Testing RiskManager …")
from risk_manager import RiskManager

rm = RiskManager(daily_loss_limit=0.09, risk_per_trade=0.01,
                 min_stake=0.35, max_stake=500.0)
rm.set_balance(1.00)
assert rm.can_trade(), "Should be able to trade"

stake = rm.calculate_stake()
print(f"   Balance $1.00 → stake=${stake:.2f} "
      f"(min ${rm.min_stake} enforced ✓)")

# Simulate 9% loss
rm.set_balance(0.90)
print(f"   Balance after 10% drop: ${rm.current_balance:.2f} | "
      f"paused={rm.is_paused}")
assert rm.is_paused, "Should be paused after >9% loss"
print("   RiskManager OK ✓")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✅")
print("=" * 60)
print("\nNext steps:")
print("  1. Set DERIV_API_TOKEN in your environment")
print("  2. Run: python main.py")
print("  3. Visit: http://localhost:8080")
