// Comparison benchmark: sbits vs sucds vs vers-vecs
//
// API notes:
//   sucds::Rank9Sel  -- built from sucds::BitVector via Rank9Sel::new(); rank/select via traits
//                       Rank::rank(pos) -> usize (count in [0,pos))
//                       Select::select(k) -> Option<usize>  (0-indexed)
//   vers_vecs::RsVec -- built via RsVec::from_bit_vec(BitVec); direct methods:
//                       rank1(pos) -> usize, rank0(pos) -> usize
//                       select1(k) -> usize (panics on OOB), select0(k) -> usize
//   sucds::EliasFano -- builder: EliasFanoBuilder::new(universe, num_vals)?.push(v)?.build()
//                       binsearch(v) for contains, successor(v), rank(v)
//   vers_vecs::EliasFano -- from_slice(&[u64]); successor(u64), predecessor(u64), rank(u64)
//
// sux is excluded: its rank/select API requires type-level parameters and epserde traits
// that make apples-to-apples benchmarking significantly more complex. Noted below.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// sucds imports
use sucds::bit_vectors::{
    prelude::{Rank, Select},
    Rank9Sel,
};
use sucds::mii_sequences::{EliasFano as SucdsEF, EliasFanoBuilder};

// vers-vecs imports
use vers_vecs::bit_vec::{fast_rs_vec::RsVec, BitVec};
use vers_vecs::EliasFanoVec as VersEF;

// sbits imports
use sbits::BitVector as SbitsBV;
use sbits::EliasFano as SbitsEF;

// ── dataset parameters ──────────────────────────────────────────────────────
const N_BITS: usize = 1_000_000;
const N_EF: usize = 100_000;
const EF_UNIVERSE: u64 = 4_000_001;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_sbits_bv() -> SbitsBV {
    let words = vec![0xAAAAAAAAAAAAAAAAu64; N_BITS / 64];
    SbitsBV::new(&words, N_BITS)
}

fn make_sucds_bv() -> Rank9Sel {
    // 50% density: alternating bits
    let bits = (0..N_BITS).map(|i| i % 2 == 1);
    Rank9Sel::from_bits(bits).select1_hints().select0_hints()
}

fn make_vers_bv() -> RsVec {
    let words = vec![0xAAAAAAAAAAAAAAAAu64; N_BITS / 64];
    let bv = BitVec::from_limbs(&words);
    RsVec::from_bit_vec(bv)
}

fn make_sbits_ef() -> SbitsEF {
    let values: Vec<u64> = (0..N_EF as u64).map(|i| i * 40).collect();
    SbitsEF::new(&values, EF_UNIVERSE)
}

fn make_sucds_ef() -> SucdsEF {
    let mut builder = EliasFanoBuilder::new(EF_UNIVERSE as usize, N_EF).unwrap();
    for i in 0..N_EF {
        builder.push(i * 40).unwrap();
    }
    // enable_rank() builds the select0 index needed for successor/predecessor/rank
    builder.build().enable_rank()
}

fn make_vers_ef() -> VersEF {
    let values: Vec<u64> = (0..N_EF as u64).map(|i| i * 40).collect();
    VersEF::from_slice(&values)
}

// ── rank1 ────────────────────────────────────────────────────────────────────

fn bench_rank1(c: &mut Criterion) {
    let sbits = make_sbits_bv();
    let sucds = make_sucds_bv();
    let vers = make_vers_bv();

    let mut group = c.benchmark_group("rank1/1M_50pct");
    let queries: Vec<usize> = (0..N_BITS).step_by(64).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &pos in &queries {
                black_box(sbits.rank1(pos));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "Rank9Sel"), &(), |b, _| {
        b.iter(|| {
            for &pos in &queries {
                // sucds Rank::rank1 returns count of 1s in [0, pos) — same semantics as sbits rank1
                black_box(sucds.rank1(pos));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &pos in &queries {
                black_box(vers.rank1(pos));
            }
        })
    });

    group.finish();
}

// ── select1 ──────────────────────────────────────────────────────────────────

fn bench_select1(c: &mut Criterion) {
    let sbits = make_sbits_bv();
    let sucds = make_sucds_bv();
    let vers = make_vers_bv();

    let total_ones = N_BITS / 2; // 50% density
    let mut group = c.benchmark_group("select1/1M_50pct");
    let queries: Vec<usize> = (0..total_ones).step_by(512).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &k in &queries {
                black_box(sbits.select1(k));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "Rank9Sel+hints"), &(), |b, _| {
        b.iter(|| {
            for &k in &queries {
                black_box(sucds.select1(k));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &k in &queries {
                // vers select1(k) panics on OOB; queries are within [0, total_ones)
                black_box(vers.select1(k));
            }
        })
    });

    group.finish();
}

// ── rank0 ─────────────────────────────────────────────────────────────────────

fn bench_rank0(c: &mut Criterion) {
    let sbits = make_sbits_bv();
    let sucds = make_sucds_bv();
    let vers = make_vers_bv();

    let mut group = c.benchmark_group("rank0/1M_50pct");
    let queries: Vec<usize> = (0..N_BITS).step_by(64).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &pos in &queries {
                black_box(sbits.rank0(pos));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "Rank9Sel"), &(), |b, _| {
        b.iter(|| {
            for &pos in &queries {
                black_box(sucds.rank0(pos));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &pos in &queries {
                black_box(vers.rank0(pos));
            }
        })
    });

    group.finish();
}

// ── select0 ──────────────────────────────────────────────────────────────────

fn bench_select0(c: &mut Criterion) {
    let sbits = make_sbits_bv();
    let sucds = make_sucds_bv();
    let vers = make_vers_bv();

    let total_zeros = N_BITS / 2;
    let mut group = c.benchmark_group("select0/1M_50pct");
    let queries: Vec<usize> = (0..total_zeros).step_by(512).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &k in &queries {
                black_box(sbits.select0(k));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "Rank9Sel+hints"), &(), |b, _| {
        b.iter(|| {
            for &k in &queries {
                black_box(sucds.select0(k));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &k in &queries {
                black_box(vers.select0(k));
            }
        })
    });

    group.finish();
}

// ── EliasFano successor ───────────────────────────────────────────────────────

fn bench_ef_successor(c: &mut Criterion) {
    let sbits_ef = make_sbits_ef();
    let sucds_ef = make_sucds_ef();
    let vers_ef = make_vers_ef();

    let mut group = c.benchmark_group("ef_successor/100K");
    let targets: Vec<u64> = (0..EF_UNIVERSE).step_by(4000).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &t in &targets {
                black_box(sbits_ef.successor(t));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "EliasFano"), &(), |b, _| {
        b.iter(|| {
            for &t in &targets {
                black_box(sucds_ef.successor(t as usize));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &t in &targets {
                black_box(vers_ef.successor(t));
            }
        })
    });

    group.finish();
}

// ── EliasFano contains/membership ────────────────────────────────────────────

fn bench_ef_contains(c: &mut Criterion) {
    let sbits_ef = make_sbits_ef();
    let sucds_ef = make_sucds_ef();
    let vers_ef = make_vers_ef();

    let mut group = c.benchmark_group("ef_contains/100K");
    // Step of 20: every other query hits an actual value (multiples of 40)
    let targets: Vec<u64> = (0..EF_UNIVERSE).step_by(20).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &t in &targets {
                black_box(sbits_ef.contains(t));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "EliasFano"), &(), |b, _| {
        b.iter(|| {
            for &t in &targets {
                // sucds: binsearch returns Some(idx) on exact match
                black_box(sucds_ef.binsearch(t as usize).is_some());
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &t in &targets {
                // vers has no direct contains; use successor equality
                black_box(vers_ef.successor(t) == Some(t));
            }
        })
    });

    group.finish();
}

// ── EliasFano rank ────────────────────────────────────────────────────────────

fn bench_ef_rank(c: &mut Criterion) {
    let sbits_ef = make_sbits_ef();
    let sucds_ef = make_sucds_ef();
    let vers_ef = make_vers_ef();

    let mut group = c.benchmark_group("ef_rank/100K");
    let targets: Vec<u64> = (0..EF_UNIVERSE).step_by(4000).collect();

    group.bench_function("sbits", |b| {
        b.iter(|| {
            for &t in &targets {
                black_box(sbits_ef.rank(t));
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("sucds", "EliasFano"), &(), |b, _| {
        b.iter(|| {
            for &t in &targets {
                black_box(sucds_ef.rank(t as usize));
            }
        })
    });

    group.bench_function("vers-vecs", |b| {
        b.iter(|| {
            for &t in &targets {
                black_box(vers_ef.rank(t));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rank1,
    bench_select1,
    bench_rank0,
    bench_select0,
    bench_ef_successor,
    bench_ef_contains,
    bench_ef_rank,
);
criterion_main!(benches);
