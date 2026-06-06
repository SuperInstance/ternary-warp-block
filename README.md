# ternary-warp-block

[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)]()

**Warp-level programming abstractions for ternary GPU kernels.**

This crate provides CPU-side simulation of GPU warp-level primitives specifically designed for ternary (trit-based: −1, 0, +1) computation. It models a warp as a collection of lanes, each holding a trit value, and implements the fundamental cooperative operations that GPU threads within a warp perform together.

## Why Ternary Warps?

Ternary neural networks and balanced ternary computing represent values in {−1, 0, +1} space. When these operations run on GPUs, warp-level cooperation is critical for performance — just as in binary CUDA programming. This crate provides the building blocks for designing and testing warp-level ternary algorithms before deploying to actual hardware.

## Core Types

### `Trit` — The Fundamental Unit

```rust
use ternary_warp_block::Trit;

let pos = Trit::Positive;   // +1
let zero = Trit::Zero;      //  0
let neg = Trit::Negative;   // -1

// Arithmetic with clamping
assert_eq!((pos + neg).to_int(), 0);        // 1 + (-1) = 0
assert_eq!((pos * neg).to_int(), -1);       // 1 × (-1) = -1
```

### `WarpConfig` — Warp Descriptor

```rust
use ternary_warp_block::{WarpConfig, Trit};

// Create a 32-lane warp with specific values
let warp = WarpConfig::new(32, vec![Trit::Positive; 32]);

// Or a zeroed warp
let zero_warp = WarpConfig::zeroed(32);
```

## Operations

### `WarpReduce` — Reductions Across Lanes

Reduce all trit values in a warp to a single result:

| Method | Description |
|--------|-------------|
| `sum` | Ternary sum (clamped to {−1, 0, +1}) |
| `sum_raw` | Full integer sum without clamping |
| `majority` | Most common trit value (democratic vote) |
| `max` / `min` | Extremal trit values |
| `all_positive` / `any_positive` | Predicate reductions |

```rust
use ternary_warp_block::{WarpConfig, Trit, WarpReduce};

let warp = WarpConfig::new(8, vec![
    Trit::Positive, Trit::Positive, Trit::Positive,
    Trit::Zero, Trit::Negative, Trit::Positive, Trit::Positive, Trit::Positive
]);

assert_eq!(WarpReduce::majority(&warp), Trit::Positive);
assert_eq!(WarpReduce::sum_raw(&warp), 5);
```

### `WarpScan` — Prefix Scans

Compute running aggregates across warp lanes:

- **Inclusive sum** — each lane contains the prefix sum up to and including itself
- **Exclusive sum** — each lane contains the prefix sum of all preceding lanes
- **Custom scan** — use any binary associative operator

```rust
use ternary_warp_block::{WarpConfig, Trit, WarpScan};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Positive, Trit::Negative, Trit::Zero
]);

let prefix = WarpScan::inclusive_sum_raw(&warp);
// [1, 2, 1, 1] — raw prefix sums before clamping
```

### `WarpShuffle` — Lane Data Exchange

Exchange trit values between lanes without shared memory:

| Method | Description |
|--------|-------------|
| `broadcast` | All lanes get value from one source lane |
| `shuffle_down` | Each lane reads from `lane + delta` |
| `shuffle_up` | Each lane reads from `lane - delta` |
| `shuffle_idx` | Arbitrary lane-to-lane mapping |
| `butterfly` | Pairwise swap (lane 2k ↔ lane 2k+1) |
| `reverse` | Reverse the lane order |

```rust
use ternary_warp_block::{WarpConfig, Trit, WarpShuffle};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Negative, Trit::Zero, Trit::Positive
]);

let reversed = WarpShuffle::reverse(&warp);
// [Positive, Zero, Negative, Positive]
```

### `WarpVote` — Collective Predicates

Test properties across all warp lanes simultaneously:

| Method | Description |
|--------|-------------|
| `all_same` | All lanes hold the same trit |
| `any_nonzero` | At least one lane is nonzero |
| `all_nonzero` | Every lane is nonzero |
| `ballot` | Bitmask of lanes matching a predicate |
| `popcount` | Count lanes matching a predicate |

```rust
use ternary_warp_block::{WarpConfig, Trit, WarpVote};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Zero, Trit::Negative, Trit::Positive
]);

assert_eq!(WarpVote::ballot(&warp, |t| t.is_nonzero()), 0b1101);
assert_eq!(WarpVote::popcount(&warp, |t| t == Trit::Positive), 2);
```

## Design Principles

- **CPU Simulation** — All operations run on CPU, making them testable and debuggable without GPU hardware
- **Configurable Warp Size** — Not locked to 32; works with any lane count
- **Ternary-Native** — Clamping semantics match ternary arithmetic, not binary emulation
- **Zero Dependencies** — Pure Rust, no external crates needed

## Testing

```bash
cargo test
```

35 comprehensive tests covering all primitives: reductions, scans, shuffles, votes, and trit arithmetic.

## License

MIT OR Apache-2.0
