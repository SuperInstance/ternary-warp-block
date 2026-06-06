# ternary-warp-block

Warp-level programming abstractions for ternary GPU kernels.

In GPU programming, a warp is the fundamental unit of parallel execution — 32 threads (on NVIDIA) that execute together in lockstep. Warp-level operations (reduce, scan, shuffle, vote) are the building blocks of efficient GPU algorithms. This crate provides those building blocks for ternary (trit-based) computation, running as CPU-side simulation so you can design, test, and verify your algorithms before deploying to hardware.

Every value in this crate is a `Trit`: one of `{-1, 0, +1}`. The arithmetic, reductions, and scans all respect ternary semantics with clamping — the output is always a valid trit.

## Why This Exists

Ternary neural networks compute with {-1, 0, +1} weights and activations. When you map these operations to GPU warps, you need collective primitives that understand ternary values:

- **Reductions**: sum a warp's trits, take majority vote, find min/max
- **Scans**: prefix sums for compression, compaction, and indexing
- **Shuffles**: exchange trits between lanes without shared memory
- **Votes**: collective predicates (all same? any nonzero? ballot mask?)

The key insight: ternary warp operations are *simpler* than their binary counterparts. A trit fits in 2 bits, so 16 trits pack into one 32-bit word. Warp-wide ternary reduction is just counting {-1, 0, +1} and classifying the result — no floating-point accumulation, no precision concerns.

## Quick Start

```rust
use ternary_warp_block::{Trit, WarpConfig, WarpReduce, WarpVote};

// Create a 32-lane warp
let warp = WarpConfig::new(32, vec![
    Trit::Positive, Trit::Negative, Trit::Positive, Trit::Zero,
    Trit::Positive, Trit::Positive, Trit::Zero, Trit::Negative,
    // ... 24 more lanes
].into_iter().cycle().take(32).collect());

// Majority vote across the warp
let majority = WarpReduce::majority(&warp);
println!("Warp votes: {:?}", majority);

// Ballot: which lanes are nonzero?
let mask = WarpVote::ballot(&warp, |t| t.is_nonzero());
println!("Nonzero lanes: {:032b}", mask);
```

## Core Types

### Trit — The Fundamental Unit

```rust
use ternary_warp_block::Trit;

let pos = Trit::Positive;   // +1
let zero = Trit::Zero;      //  0
let neg = Trit::Negative;   // -1

// Arithmetic with clamping to ternary range
assert_eq!((pos + neg).to_int(), 0);        // 1 + (-1) = 0 ✓
assert_eq!((pos + pos).to_int(), 1);        // 1 + 1 = 2, clamped to 1
assert_eq!((pos * neg).to_int(), -1);       // 1 × (-1) = -1
assert_eq!((-neg).to_int(), 1);             // negation
```

The clamping is deliberate: ternary arithmetic always produces ternary results. This matches hardware behavior where trit operations saturate rather than overflow.

### WarpConfig — The Warp Descriptor

```rust
use ternary_warp_block::{WarpConfig, Trit};

// Explicit values
let warp = WarpConfig::new(8, vec![
    Trit::Positive, Trit::Negative, Trit::Zero, Trit::Positive,
    Trit::Negative, Trit::Zero, Trit::Positive, Trit::Negative,
]);

// Zeroed warp (useful for initializing output buffers)
let empty = WarpConfig::zeroed(32);
```

Warp size is configurable — not locked to 32. This lets you simulate sub-warps (4, 8, 16 lanes) for testing or target architectures with different warp widths.

## Operations

### WarpReduce — Reductions Across Lanes

```rust
use ternary_warp_block::{WarpReduce, WarpConfig, Trit};

let warp = WarpConfig::new(8, vec![
    Trit::Positive, Trit::Positive, Trit::Positive,
    Trit::Zero, Trit::Negative, Trit::Positive, Trit::Positive, Trit::Positive,
]);

// Ternary sum (clamped): 6 positive, 1 zero, 1 negative → raw sum = 5, clamped = +1
assert_eq!(WarpReduce::sum(&warp), Trit::Positive);
assert_eq!(WarpReduce::sum_raw(&warp), 5);

// Majority vote: most common trit
assert_eq!(WarpReduce::majority(&warp), Trit::Positive);

// Extremes
assert_eq!(WarpReduce::max(&warp), Trit::Positive);
assert_eq!(WarpReduce::min(&warp), Trit::Negative);

// Predicate reductions
assert!(WarpReduce::all_positive(&warp) == false);
assert!(WarpReduce::any_positive(&warp) == true);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `sum` | `Trit` | Clamped ternary sum |
| `sum_raw` | `i64` | Unclamped integer sum |
| `majority` | `Trit` | Most common value (democratic vote) |
| `max` | `Trit` | Maximum value (Positive > Zero > Negative) |
| `min` | `Trit` | Minimum value |
| `all_positive` | `bool` | Every lane is Positive? |
| `any_positive` | `bool` | Any lane is Positive? |

### WarpScan — Prefix Scans

```rust
use ternary_warp_block::{WarpScan, WarpConfig, Trit};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Positive, Trit::Negative, Trit::Zero,
]);

// Inclusive prefix sum (raw): [1, 2, 1, 1]
let raw = WarpScan::inclusive_sum_raw(&warp);
assert_eq!(raw, vec![1, 2, 1, 1]);

// Inclusive prefix sum (clamped to trits): [1, 1, 1, 1]
let clamped = WarpScan::inclusive_sum(&warp);

// Exclusive: lane i contains sum of lanes 0..i
let excl = WarpScan::exclusive_sum(&warp);
// [Zero, Positive, Positive, Positive]

// Custom scan with any binary operator
let product_scan = WarpScan::inclusive_scan(&warp, |a, b| a * b);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `inclusive_sum` | `Vec<Trit>` | Clamped prefix sum |
| `inclusive_sum_raw` | `Vec<i64>` | Unclamped prefix sum |
| `exclusive_sum` | `Vec<Trit>` | Exclusive prefix sum |
| `inclusive_scan(op)` | `Vec<Trit>` | Custom binary operator scan |

### WarpShuffle — Lane Data Exchange

```rust
use ternary_warp_block::{WarpShuffle, WarpConfig, Trit};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Negative, Trit::Zero, Trit::Positive,
]);

// Broadcast: all lanes get value from lane 1
let all_neg = WarpShuffle::broadcast(&warp, 1);
assert_eq!(all_neg, vec![Trit::Negative; 4]);

// Rotate down by 1: each lane reads from (lane + 1) % 4
let rotated = WarpShuffle::shuffle_down(&warp, 1);
// [Negative, Zero, Positive, Positive]

// Reverse
let rev = WarpShuffle::reverse(&warp);
// [Positive, Zero, Negative, Positive]

// Butterfly (pairwise swap): lanes 0↔1, 2↔3
let bfly = WarpShuffle::butterfly(&warp);
// [Negative, Positive, Positive, Zero]

// Arbitrary index mapping
let mapped = WarpShuffle::shuffle_idx(&warp, &[3, 2, 1, 0]);
```

| Method | Description |
|--------|-------------|
| `broadcast(src)` | All lanes get `lanes[src]` |
| `shuffle_down(delta)` | Rotate values down by delta |
| `shuffle_up(delta)` | Rotate values up by delta |
| `shuffle_idx(map)` | Arbitrary lane-to-lane mapping |
| `butterfly()` | Pairwise swap (0↔1, 2↔3, ...) |
| `reverse()` | Reverse lane order |

### WarpVote — Collective Predicates

```rust
use ternary_warp_block::{WarpVote, WarpConfig, Trit};

let warp = WarpConfig::new(4, vec![
    Trit::Positive, Trit::Zero, Trit::Negative, Trit::Positive,
]);

assert!(!WarpVote::all_same(&warp));
assert!(WarpVote::any_nonzero(&warp));

// Ballot: bitmask of lanes matching predicate
let nonzero_mask = WarpVote::ballot(&warp, |t| t.is_nonzero());
assert_eq!(nonzero_mask, 0b1101); // lanes 0, 2, 3

let positive_count = WarpVote::popcount(&warp, |t| t == Trit::Positive);
assert_eq!(positive_count, 2);
```

| Method | Returns | Description |
|--------|---------|-------------|
| `all_same` | `bool` | All lanes have identical value |
| `any_nonzero` | `bool` | At least one nonzero lane |
| `all_nonzero` | `bool` | Every lane is nonzero |
| `ballot(pred)` | `u64` | Bitmask of matching lanes |
| `popcount(pred)` | `usize` | Count of matching lanes |

## Real-World Example: Ternary Activation Compression

```rust
use ternary_warp_block::*;

// Simulate a warp processing 32 ternary activations
let activations: Vec<Trit> = (0..32)
    .map(|i| match i % 3 {
        0 => Trit::Positive,
        1 => Trit::Negative,
        _ => Trit::Zero,
    })
    .collect();
let warp = WarpConfig::new(32, activations);

// Step 1: Majority vote — what's the dominant activation?
let consensus = WarpReduce::majority(&warp);

// Step 2: How many activations match the consensus?
let agreement = WarpVote::popcount(&warp, |t| t == consensus);
println!("{}/32 lanes agree on {:?}", agreement, consensus);

// Step 3: Ballot mask for nonzero activations (useful for sparsity)
let nonzero_mask = WarpVote::ballot(&warp, |t| t.is_nonzero());
println!("Nonzero pattern: {:032b} ({} zeros)",
    nonzero_mask, 32 - WarpVote::popcount(&warp, |t| t.is_nonzero()));

// Step 4: Prefix sum of nonzero activations for compaction
let compact = WarpScan::inclusive_sum_raw(&warp);
// Use these indices to pack nonzero trits into a dense array
```

## Ecosystem Connections

- **`ternary-grid-launch`** — Determines how many warps to launch; each warp uses these primitives
- **`ternary-shared-memory`** — Shared memory tiles that warps cooperatively load and process
- **`ternary-register-file`** — Register allocation for values produced by warp operations
- **`ternary-fence`** — Signal between warps when cooperative work is done

## Design Principles

- **CPU simulation**: Everything runs on CPU. Test and debug without a GPU.
- **Configurable warp size**: Not locked to 32. Works with any lane count.
- **Ternary-native clamping**: Arithmetic always produces valid trits, matching hardware saturation semantics.
- **Zero dependencies**: Pure Rust, no external crates.

## Performance Notes

- **Reductions**: O(warp_size) scan. On real hardware, these map to `__reduce_add_sync` and `__ballot_sync` intrinsics.
- **Scans**: O(warp_size) sequential. Hardware does this in O(log warp_size) via the Kogge-Stone algorithm. The CPU simulation is for correctness, not speed.
- **Shuffles**: O(warp_size) to produce the output vector. On hardware, these are single-instruction operations (`__shfl_sync`).
- **Votes**: O(warp_size) for the simulation. On hardware, `__all_sync`, `__any_sync`, `__ballot_sync` are single instructions.

## Open Questions

- **SIMD acceleration**: The CPU simulation could use `std::simd` for batch trit operations. Currently scalar.
- **Kogge-Stone scan**: The inclusive scan is sequential. A parallel Kogge-Stone implementation would be more representative of hardware behavior.
- **Warp matrix operations**: No warp-level matrix multiply (mma). This would be needed for ternary GEMM simulation.

## License

MIT OR Apache-2.0
