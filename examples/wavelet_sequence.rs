/// Rank/select over a sequence with a small alphabet.
///
/// A wavelet tree answers three queries on a fixed sequence without
/// scanning it: access(i) returns the symbol at position i, rank(s, i)
/// counts occurrences of symbol s in the first i positions, and
/// select(s, k) returns the position of the k-th occurrence of s.
///
/// ```sh
/// cargo run --example wavelet_sequence
/// ```
use sbits::WaveletTree;

fn main() {
    // A stream of log-level codes: 0 = INFO, 1 = WARN, 2 = ERROR.
    let levels = [0u32, 0, 2, 1, 0, 2, 2, 1, 0, 2];
    let sigma = 3; // alphabet size: every code is in 0..sigma
    let wt = WaveletTree::new(&levels, sigma);

    println!("sequence length: {}", wt.len());

    // access: what symbol sits at position 2?
    println!("levels[2] = {} (2 = ERROR)", wt.access(2));
    assert_eq!(wt.access(2), 2);

    // rank: how many ERROR codes in the first 7 entries?
    let errors_in_prefix = wt.rank(2, 7);
    println!("ERROR count in [0, 7) = {}", errors_in_prefix);
    assert_eq!(errors_in_prefix, 3);

    // select: position of the first ERROR.
    println!("first ERROR at index {:?}", wt.select(2, 0));
    assert_eq!(wt.select(2, 0), Some(2));

    // total ERROR codes across the whole sequence.
    let total_errors = wt.rank(2, wt.len());
    println!("total ERROR count = {}", total_errors);
    assert_eq!(total_errors, 4);
}
