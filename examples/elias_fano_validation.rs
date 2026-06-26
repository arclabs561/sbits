//! Validate Elias-Fano against naive references on random monotone sequences.
//!
//! Elias-Fano splits each value into high/low bits and stores the highs in a
//! bitvector navigated by rank/select; that split plus the select0 bucket walk
//! is where off-by-one bugs hide. This builds random sorted sequences and checks
//! every operation against an O(n) naive oracle: `get(i) == values[i]`,
//! `rank(t)` == count strictly less than `t`, `predecessor(t)` == largest <= t,
//! `next_geq(t)` == smallest >= t. Targets cover each value and its neighbors,
//! the boundaries, and randoms.
//!
//! ```sh
//! cargo run --release --example elias_fano_validation
//! ```

use std::process::ExitCode;

use sbits::EliasFano;

/// Deterministic strictly-increasing sequence of `n` values within `universe`.
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
        cur += 1 + (next() % 50); // gap 1..=50, strictly increasing
        if cur >= universe {
            break;
        }
        v.push(cur);
    }
    v
}

fn main() -> ExitCode {
    let mut failures = 0u64;
    let mut checks = 0u64;
    let report = |cond: bool, what: &str, failures: &mut u64, checks: &mut u64| {
        *checks += 1;
        if !cond {
            *failures += 1;
            if *failures <= 5 {
                eprintln!("  VIOLATION: {what}");
            }
        }
    };

    for (n, seed) in [(200usize, 1u64), (1000, 7), (5000, 42)] {
        let universe = 4 * n as u64 * 50;
        let values = monotone_seq(n, universe, seed);
        let ef = EliasFano::new(&values, universe);

        // get(i) == values[i]
        for (i, &want) in values.iter().enumerate() {
            report(
                ef.get(i).map(|g| g == want).unwrap_or(false),
                &format!("get({i}) != {want}"),
                &mut failures,
                &mut checks,
            );
        }

        // Build the target battery: each value +/-1, boundaries, randoms.
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
            // rank: strictly less than t
            let naive_rank = values.iter().filter(|&&v| v < t).count();
            report(
                ef.rank(t) == naive_rank,
                &format!("rank({t})"),
                &mut failures,
                &mut checks,
            );

            // predecessor: largest value <= t
            let naive_pred = values.iter().copied().filter(|&v| v <= t).max();
            report(
                ef.predecessor(t) == naive_pred,
                &format!("predecessor({t})"),
                &mut failures,
                &mut checks,
            );

            // next_geq: smallest value >= t
            let naive_succ = values.iter().copied().find(|&v| v >= t);
            report(
                ef.next_geq(t) == naive_succ,
                &format!("next_geq({t})"),
                &mut failures,
                &mut checks,
            );
        }

        println!(
            "n={n:<5} universe={universe:<7} values={:<5} checks so far={checks}",
            values.len()
        );
    }

    println!("\n{checks} checks, {failures} violations");
    if failures == 0 {
        println!("PASS: Elias-Fano matches the naive oracle on all operations");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL: Elias-Fano diverged from the naive oracle");
        ExitCode::FAILURE
    }
}
