//! Elias-Fano operations vs an O(n) naive oracle on random monotone sequences.
//!
//! The high/low bit split plus the select0 bucket walk is where off-by-one bugs
//! hide. For random sorted sequences, every operation is checked against a naive
//! reference: get(i)==values[i], rank(t)==#{v<t}, predecessor(t)==max{v<=t},
//! next_geq(t)==min{v>=t}, over targets covering each value, its neighbors, the
//! boundaries, and randoms.

use sbits::EliasFano;

fn monotone_seq(n: usize, universe: u64, seed: u64) -> Vec<u64> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut v = Vec::with_capacity(n);
    let mut cur = 0u64;
    for _ in 0..n {
        cur += 1 + (next() % 50);
        if cur >= universe {
            break;
        }
        v.push(cur);
    }
    v
}

#[test]
fn elias_fano_matches_naive_oracle() {
    for (n, seed) in [(200usize, 1u64), (1000, 7), (5000, 42)] {
        let universe = 4 * n as u64 * 50;
        let values = monotone_seq(n, universe, seed);
        let ef = EliasFano::new(&values, universe);

        for (i, &want) in values.iter().enumerate() {
            assert_eq!(ef.get(i).expect("get"), want, "get({i}) != {want} (n={n})");
        }

        let mut targets: Vec<u64> = Vec::new();
        for &v in &values {
            targets.push(v);
            targets.push(v.saturating_sub(1));
            targets.push(v + 1);
        }
        targets.extend([0, universe.saturating_sub(1), universe]);
        let mut s = seed.wrapping_mul(2654435761) | 1;
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            targets.push(s % (universe + 1));
        }

        for &t in &targets {
            let naive_rank = values.iter().filter(|&&v| v < t).count();
            assert_eq!(ef.rank(t), naive_rank, "rank({t}) (n={n})");
            let naive_pred = values.iter().copied().filter(|&v| v <= t).max();
            assert_eq!(ef.predecessor(t), naive_pred, "predecessor({t}) (n={n})");
            let naive_succ = values.iter().copied().find(|&v| v >= t);
            assert_eq!(ef.next_geq(t), naive_succ, "next_geq({t}) (n={n})");
        }
    }
}
