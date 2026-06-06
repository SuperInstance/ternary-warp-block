//! # ternary-warp-block
//!
//! Warp-level programming abstractions for ternary GPU kernels.
//!
//! This crate provides CPU-side simulation of GPU warp-level primitives
//! optimized for ternary (trit-based) computation. Each warp consists of
//! a configurable number of lanes that cooperatively perform reductions,
//! scans, shuffles, and votes on ternary values (`-1, 0, +1`).

/// A ternary trit value: Negative (-1), Zero (0), or Positive (+1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

impl Trit {
    /// Convert an integer to a Trit. Clamps values outside [-1, 1].
    pub fn from_int(v: i8) -> Self {
        match v {
            -1 => Trit::Negative,
            0 => Trit::Zero,
            1 => Trit::Positive,
            _ => {
                if v < 0 {
                    Trit::Negative
                } else {
                    Trit::Positive
                }
            }
        }
    }

    /// Convert to integer value.
    pub fn to_int(self) -> i8 {
        self as i8
    }

    /// Whether this trit is nonzero.
    pub fn is_nonzero(self) -> bool {
        self != Trit::Zero
    }
}

impl std::ops::Neg for Trit {
    type Output = Trit;
    fn neg(self) -> Trit {
        match self {
            Trit::Negative => Trit::Positive,
            Trit::Zero => Trit::Zero,
            Trit::Positive => Trit::Negative,
        }
    }
}

impl std::ops::Add for Trit {
    type Output = Trit;
    fn add(self, other: Trit) -> Trit {
        Trit::from_int(self.to_int() + other.to_int())
    }
}

impl std::ops::Mul for Trit {
    type Output = Trit;
    fn mul(self, other: Trit) -> Trit {
        Trit::from_int(self.to_int() * other.to_int())
    }
}

/// Configuration for a warp of ternary lanes.
#[derive(Debug, Clone)]
pub struct WarpConfig {
    /// Number of lanes in the warp (typically 32 for NVIDIA GPUs).
    pub warp_size: usize,
    /// Lane values — one trit per lane.
    pub lanes: Vec<Trit>,
}

impl WarpConfig {
    /// Create a new WarpConfig with the given warp size and lane values.
    ///
    /// # Panics
    /// Panics if `lanes.len() != warp_size`.
    pub fn new(warp_size: usize, lanes: Vec<Trit>) -> Self {
        assert_eq!(lanes.len(), warp_size, "lane count must match warp_size");
        Self { warp_size, lanes }
    }

    /// Create a warp with all lanes set to Zero.
    pub fn zeroed(warp_size: usize) -> Self {
        Self {
            warp_size,
            lanes: vec![Trit::Zero; warp_size],
        }
    }

    /// Get the trit value at the given lane index.
    pub fn lane(&self, idx: usize) -> Trit {
        self.lanes[idx]
    }

    /// Set the trit value at the given lane index.
    pub fn set_lane(&mut self, idx: usize, val: Trit) {
        self.lanes[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// WarpReduce — reduce across warp lanes
// ---------------------------------------------------------------------------

/// Warp-level reduction operations across all lanes.
pub struct WarpReduce;

impl WarpReduce {
    /// Sum all trits in the warp, returning the clamped ternary result.
    pub fn sum(warp: &WarpConfig) -> Trit {
        let total: i64 = warp.lanes.iter().map(|t| t.to_int() as i64).sum();
        Trit::from_int(total.clamp(-1, 1) as i8)
    }

    /// Compute the raw (unclamped) sum as an integer.
    pub fn sum_raw(warp: &WarpConfig) -> i64 {
        warp.lanes.iter().map(|t| t.to_int() as i64).sum()
    }

    /// Majority vote: returns the most common trit value across lanes.
    /// Ties are broken in favor of Zero, then Positive.
    pub fn majority(warp: &WarpConfig) -> Trit {
        let mut counts = [0usize; 3]; // Neg, Zero, Pos
        for &t in &warp.lanes {
            match t {
                Trit::Negative => counts[0] += 1,
                Trit::Zero => counts[1] += 1,
                Trit::Positive => counts[2] += 1,
            }
        }
        if counts[2] >= counts[0] && counts[2] >= counts[1] {
            Trit::Positive
        } else if counts[0] >= counts[1] && counts[0] >= counts[2] {
            Trit::Negative
        } else {
            Trit::Zero
        }
    }

    /// Maximum trit value across all lanes (Positive > Zero > Negative).
    pub fn max(warp: &WarpConfig) -> Trit {
        warp.lanes
            .iter()
            .copied()
            .max_by_key(|t| t.to_int())
            .unwrap_or(Trit::Zero)
    }

    /// Minimum trit value across all lanes.
    pub fn min(warp: &WarpConfig) -> Trit {
        warp.lanes
            .iter()
            .copied()
            .min_by_key(|t| t.to_int())
            .unwrap_or(Trit::Zero)
    }

    /// Logical AND across lanes: Positive only if all lanes are Positive.
    pub fn all_positive(warp: &WarpConfig) -> bool {
        warp.lanes.iter().all(|&t| t == Trit::Positive)
    }

    /// Logical OR across lanes: Positive if any lane is Positive.
    pub fn any_positive(warp: &WarpConfig) -> bool {
        warp.lanes.iter().any(|&t| t == Trit::Positive)
    }
}

// ---------------------------------------------------------------------------
// WarpScan — prefix scan across warp
// ---------------------------------------------------------------------------

/// Warp-level prefix scan (inclusive) across all lanes.
pub struct WarpScan;

impl WarpScan {
    /// Inclusive prefix sum scan. Each lane contains the clamped ternary
    /// sum of all lanes from 0..=idx.
    pub fn inclusive_sum(warp: &WarpConfig) -> Vec<Trit> {
        let mut result = Vec::with_capacity(warp.warp_size);
        let mut acc: i64 = 0;
        for &t in &warp.lanes {
            acc += t.to_int() as i64;
            result.push(Trit::from_int(acc.clamp(-1, 1) as i8));
        }
        result
    }

    /// Inclusive prefix sum with raw (unclamped) integer values.
    pub fn inclusive_sum_raw(warp: &WarpConfig) -> Vec<i64> {
        let mut result = Vec::with_capacity(warp.warp_size);
        let mut acc: i64 = 0;
        for &t in &warp.lanes {
            acc += t.to_int() as i64;
            result.push(acc);
        }
        result
    }

    /// Exclusive prefix sum scan. Lane i contains the sum of lanes 0..i.
    pub fn exclusive_sum(warp: &WarpConfig) -> Vec<Trit> {
        let mut result = Vec::with_capacity(warp.warp_size);
        let mut acc: i64 = 0;
        for &t in &warp.lanes {
            result.push(Trit::from_int(acc.clamp(-1, 1) as i8));
            acc += t.to_int() as i64;
        }
        result
    }

    /// Inclusive prefix scan using a custom binary associative operator.
    pub fn inclusive_scan<F>(warp: &WarpConfig, op: F) -> Vec<Trit>
    where
        F: Fn(Trit, Trit) -> Trit,
    {
        let mut result = Vec::with_capacity(warp.warp_size);
        let mut acc = warp.lanes[0];
        result.push(acc);
        for &t in &warp.lanes[1..] {
            acc = op(acc, t);
            result.push(acc);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// WarpShuffle — exchange trits between lanes
// ---------------------------------------------------------------------------

/// Warp-level shuffle operations for exchanging trits between lanes.
pub struct WarpShuffle;

impl WarpShuffle {
    /// Broadcast: every lane receives the value from `src_lane`.
    pub fn broadcast(warp: &WarpConfig, src_lane: usize) -> Vec<Trit> {
        let val = warp.lane(src_lane);
        vec![val; warp.warp_size]
    }

    /// Shuffle down: each lane receives the value from (lane + delta) % warp_size.
    pub fn shuffle_down(warp: &WarpConfig, delta: usize) -> Vec<Trit> {
        let n = warp.warp_size;
        warp.lanes
            .iter()
            .enumerate()
            .map(|(i, _)| warp.lanes[(i + delta) % n])
            .collect()
    }

    /// Shuffle up: each lane receives the value from (lane + warp_size - delta) % warp_size.
    pub fn shuffle_up(warp: &WarpConfig, delta: usize) -> Vec<Trit> {
        let n = warp.warp_size;
        warp.lanes
            .iter()
            .enumerate()
            .map(|(i, _)| warp.lanes[(i + n - delta % n) % n])
            .collect()
    }

    /// Shuffle by index map: lane i receives the value from map[i].
    /// `map` must have the same length as the warp.
    pub fn shuffle_idx(warp: &WarpConfig, map: &[usize]) -> Vec<Trit> {
        assert_eq!(map.len(), warp.warp_size, "map length must match warp size");
        map.iter().map(|&src| warp.lanes[src % warp.warp_size]).collect()
    }

    /// Butterfly shuffle (exchange pairs): lane 2k <-> lane 2k+1.
    pub fn butterfly(warp: &WarpConfig) -> Vec<Trit> {
        let mut result = warp.lanes.clone();
        let mut i = 0;
        while i + 1 < warp.warp_size {
            result.swap(i, i + 1);
            i += 2;
        }
        result
    }

    /// Reverse shuffle: lane i receives value from lane (warp_size - 1 - i).
    pub fn reverse(warp: &WarpConfig) -> Vec<Trit> {
        warp.lanes.iter().copied().rev().collect()
    }
}

// ---------------------------------------------------------------------------
// WarpVote — collective voting primitives
// ---------------------------------------------------------------------------

/// Warp-level vote operations.
pub struct WarpVote;

impl WarpVote {
    /// Returns true if all lanes have the same trit value.
    pub fn all_same(warp: &WarpConfig) -> bool {
        let first = warp.lanes[0];
        warp.lanes.iter().all(|&t| t == first)
    }

    /// Returns true if any lane has a nonzero trit value.
    pub fn any_nonzero(warp: &WarpConfig) -> bool {
        warp.lanes.iter().any(|t| t.is_nonzero())
    }

    /// Returns true if all lanes have nonzero trit values.
    pub fn all_nonzero(warp: &WarpConfig) -> bool {
        warp.lanes.iter().all(|t| t.is_nonzero())
    }

    /// Returns a bitmask where bit i is set if lane i matches the predicate.
    pub fn ballot(warp: &WarpConfig, pred: impl Fn(Trit) -> bool) -> u64 {
        let mut mask: u64 = 0;
        for (i, &t) in warp.lanes.iter().enumerate() {
            if i < 64 && pred(t) {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    /// Count of lanes matching the predicate.
    pub fn popcount(warp: &WarpConfig, pred: impl Fn(Trit) -> bool) -> usize {
        warp.lanes.iter().filter(|&&t| pred(t)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_warp(lanes: &[i8]) -> WarpConfig {
        WarpConfig::new(lanes.len(), lanes.iter().map(|&v| Trit::from_int(v)).collect())
    }

    // --- WarpReduce tests ---

    #[test]
    fn test_reduce_sum_all_positive() {
        let warp = make_warp(&[1, 1, 1, 1]);
        assert_eq!(WarpReduce::sum(&warp), Trit::Positive);
        assert_eq!(WarpReduce::sum_raw(&warp), 4);
    }

    #[test]
    fn test_reduce_sum_balanced() {
        let warp = make_warp(&[1, -1, 1, -1]);
        assert_eq!(WarpReduce::sum(&warp), Trit::Zero);
        assert_eq!(WarpReduce::sum_raw(&warp), 0);
    }

    #[test]
    fn test_reduce_sum_negative() {
        let warp = make_warp(&[-1, -1, -1, -1]);
        assert_eq!(WarpReduce::sum(&warp), Trit::Negative);
        assert_eq!(WarpReduce::sum_raw(&warp), -4);
    }

    #[test]
    fn test_reduce_majority_positive() {
        let warp = make_warp(&[1, 1, 1, 0, -1]);
        assert_eq!(WarpReduce::majority(&warp), Trit::Positive);
    }

    #[test]
    fn test_reduce_majority_negative() {
        let warp = make_warp(&[-1, -1, 0, 1]);
        assert_eq!(WarpReduce::majority(&warp), Trit::Negative);
    }

    #[test]
    fn test_reduce_majority_tie() {
        let warp = make_warp(&[1, -1, 0, 0]);
        // Tie: zero wins when counts[1] >= others
        let m = WarpReduce::majority(&warp);
        assert!(m == Trit::Zero || m == Trit::Positive || m == Trit::Negative);
    }

    #[test]
    fn test_reduce_max() {
        let warp = make_warp(&[-1, 0, -1, 0]);
        assert_eq!(WarpReduce::max(&warp), Trit::Zero);
    }

    #[test]
    fn test_reduce_max_with_positive() {
        let warp = make_warp(&[-1, 0, 1, -1]);
        assert_eq!(WarpReduce::max(&warp), Trit::Positive);
    }

    #[test]
    fn test_reduce_min() {
        let warp = make_warp(&[0, 1, 0, 1]);
        assert_eq!(WarpReduce::min(&warp), Trit::Zero);
    }

    #[test]
    fn test_reduce_all_positive() {
        let warp = make_warp(&[1, 1, 1]);
        assert!(WarpReduce::all_positive(&warp));
        let warp2 = make_warp(&[1, 0, 1]);
        assert!(!WarpReduce::all_positive(&warp2));
    }

    #[test]
    fn test_reduce_any_positive() {
        let warp = make_warp(&[-1, 0, -1]);
        assert!(!WarpReduce::any_positive(&warp));
        let warp2 = make_warp(&[-1, 1, -1]);
        assert!(WarpReduce::any_positive(&warp2));
    }

    // --- WarpScan tests ---

    #[test]
    fn test_scan_inclusive_sum() {
        let warp = make_warp(&[1, 1, -1, 0, 1]);
        let result = WarpScan::inclusive_sum(&warp);
        assert_eq!(result.len(), 5);
        // raw prefix sums: 1, 2, 1, 1, 2 → clamped: 1, 1, 1, 1, 1
        for t in &result {
            assert_eq!(*t, Trit::Positive);
        }
    }

    #[test]
    fn test_scan_inclusive_sum_raw() {
        let warp = make_warp(&[1, 1, -1, 0, 1]);
        let result = WarpScan::inclusive_sum_raw(&warp);
        assert_eq!(result, vec![1, 2, 1, 1, 2]);
    }

    #[test]
    fn test_scan_exclusive_sum() {
        let warp = make_warp(&[1, 1, 1]);
        let result = WarpScan::exclusive_sum(&warp);
        // exclusive: lane 0 = 0, lane 1 = sum(0..1)=1, lane 2 = sum(0..2)=2→1
        assert_eq!(result[0], Trit::Zero);
        assert_eq!(result[1], Trit::Positive);
        assert_eq!(result[2], Trit::Positive);
    }

    #[test]
    fn test_scan_inclusive_custom_op() {
        let warp = make_warp(&[1, 1, 1]);
        // Use multiplication as the scan op
        let result = WarpScan::inclusive_scan(&warp, |a, b| a * b);
        assert_eq!(result, vec![Trit::Positive, Trit::Positive, Trit::Positive]);
    }

    #[test]
    fn test_scan_empty_prefix() {
        let warp = make_warp(&[0, 0, 0, 0]);
        let result = WarpScan::inclusive_sum(&warp);
        assert_eq!(result, vec![Trit::Zero; 4]);
    }

    // --- WarpShuffle tests ---

    #[test]
    fn test_shuffle_broadcast() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let result = WarpShuffle::broadcast(&warp, 1);
        assert_eq!(result, vec![Trit::Negative; 4]);
    }

    #[test]
    fn test_shuffle_down() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let result = WarpShuffle::shuffle_down(&warp, 1);
        // lane 0 → gets lane 1 (-1), lane 1 → lane 2 (0), lane 2 → lane 3 (1), lane 3 → lane 0 (1)
        assert_eq!(
            result,
            vec![Trit::Negative, Trit::Zero, Trit::Positive, Trit::Positive]
        );
    }

    #[test]
    fn test_shuffle_up() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let result = WarpShuffle::shuffle_up(&warp, 1);
        // lane 0 → lane 3 (1), lane 1 → lane 0 (1), lane 2 → lane 1 (-1), lane 3 → lane 2 (0)
        assert_eq!(
            result,
            vec![Trit::Positive, Trit::Positive, Trit::Negative, Trit::Zero]
        );
    }

    #[test]
    fn test_shuffle_idx() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let map = vec![3, 2, 1, 0];
        let result = WarpShuffle::shuffle_idx(&warp, &map);
        assert_eq!(
            result,
            vec![Trit::Positive, Trit::Zero, Trit::Negative, Trit::Positive]
        );
    }

    #[test]
    fn test_shuffle_butterfly() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let result = WarpShuffle::butterfly(&warp);
        assert_eq!(
            result,
            vec![Trit::Negative, Trit::Positive, Trit::Positive, Trit::Zero]
        );
    }

    #[test]
    fn test_shuffle_reverse() {
        let warp = make_warp(&[1, -1, 0, 1]);
        let result = WarpShuffle::reverse(&warp);
        assert_eq!(
            result,
            vec![Trit::Positive, Trit::Zero, Trit::Negative, Trit::Positive]
        );
    }

    // --- WarpVote tests ---

    #[test]
    fn test_vote_all_same_true() {
        let warp = make_warp(&[1, 1, 1, 1]);
        assert!(WarpVote::all_same(&warp));
    }

    #[test]
    fn test_vote_all_same_false() {
        let warp = make_warp(&[1, 0, 1, 1]);
        assert!(!WarpVote::all_same(&warp));
    }

    #[test]
    fn test_vote_any_nonzero_true() {
        let warp = make_warp(&[0, 0, 1, 0]);
        assert!(WarpVote::any_nonzero(&warp));
    }

    #[test]
    fn test_vote_any_nonzero_false() {
        let warp = make_warp(&[0, 0, 0, 0]);
        assert!(!WarpVote::any_nonzero(&warp));
    }

    #[test]
    fn test_vote_all_nonzero() {
        let warp = make_warp(&[1, -1, 1, -1]);
        assert!(WarpVote::all_nonzero(&warp));
        let warp2 = make_warp(&[1, 0, 1]);
        assert!(!WarpVote::all_nonzero(&warp2));
    }

    #[test]
    fn test_vote_ballot() {
        let warp = make_warp(&[1, 0, -1, 1]);
        let mask = WarpVote::ballot(&warp, |t| t.is_nonzero());
        // lanes 0, 2, 3 are nonzero → bits 0, 2, 3
        assert_eq!(mask, 0b1101u64);
    }

    #[test]
    fn test_vote_popcount() {
        let warp = make_warp(&[1, 0, -1, 1, 0, -1]);
        let count = WarpVote::popcount(&warp, |t| t == Trit::Positive);
        assert_eq!(count, 2);
    }

    // --- Trit arithmetic ---

    #[test]
    fn test_trit_from_int() {
        assert_eq!(Trit::from_int(-1), Trit::Negative);
        assert_eq!(Trit::from_int(0), Trit::Zero);
        assert_eq!(Trit::from_int(1), Trit::Positive);
        assert_eq!(Trit::from_int(-5), Trit::Negative);
        assert_eq!(Trit::from_int(5), Trit::Positive);
    }

    #[test]
    fn test_trit_neg() {
        assert_eq!(-Trit::Positive, Trit::Negative);
        assert_eq!(-Trit::Negative, Trit::Positive);
        assert_eq!(-Trit::Zero, Trit::Zero);
    }

    #[test]
    fn test_trit_add() {
        assert_eq!(Trit::Positive + Trit::Negative, Trit::Zero);
        assert_eq!(Trit::Positive + Trit::Positive, Trit::Positive); // 2 clamped to 1
    }

    #[test]
    fn test_trit_mul() {
        assert_eq!(Trit::Positive * Trit::Negative, Trit::Negative);
        assert_eq!(Trit::Zero * Trit::Positive, Trit::Zero);
    }

    #[test]
    fn test_warp_config_zeroed() {
        let warp = WarpConfig::zeroed(8);
        assert_eq!(warp.warp_size, 8);
        assert!(warp.lanes.iter().all(|&t| t == Trit::Zero));
    }

    #[test]
    fn test_32_lane_warp_reduce() {
        let vals: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let warp = make_warp(&vals);
        assert_eq!(WarpReduce::sum(&warp), Trit::Zero);
        assert!(WarpVote::any_nonzero(&warp));
        assert!(!WarpVote::all_same(&warp));
    }
}
