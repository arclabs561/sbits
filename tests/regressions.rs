//! Regression tests for query results that disagreed with a naive oracle.

use sbits::WaveletTree;

/// `rank(symbol, i)` counts occurrences in `[0, i)`; an `i` past the end
/// counts the whole sequence, for every symbol.
#[test]
fn wavelet_rank_past_len_counts_every_occurrence() {
    let data = [0u32, 1, 1, 0, 2];
    let wt = WaveletTree::new(&data, 3);
    for s in 0..3 {
        let total = data.iter().filter(|&&x| x == s).count();
        assert_eq!(wt.rank(s, data.len()), total, "symbol {s} at len");
        assert_eq!(wt.rank(s, 100), total, "symbol {s} past len");
    }
}
