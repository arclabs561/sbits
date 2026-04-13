use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sbits::bitvec::BitVector;
use sbits::elias_fano::EliasFano;
use sbits::partitioned_elias_fano::PartitionedEliasFano;
use sbits::wavelet::WaveletTree;

fn bench_bitvector(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitvector");
    let bits = vec![0xAAAAAAAAAAAAAAAAu64; 1000]; // 64000 bits, 50% density
    let bv = BitVector::new(&bits, 64000);

    group.bench_function("rank1", |b| {
        b.iter(|| {
            for i in (0..64000).step_by(4) {
                black_box(bv.rank1(i));
            }
        })
    });

    group.bench_function("select1", |b| {
        b.iter(|| {
            for k in (0..32000).step_by(4) {
                black_box(bv.select1(k));
            }
        })
    });

    group.bench_function("ones_iter", |b| {
        b.iter(|| {
            black_box(bv.ones().count());
        })
    });

    // Large bitvector: 1M bits
    let large_bits = vec![0xAAAAAAAAAAAAAAAAu64; 1_000_000 / 64];
    let large_bv = BitVector::new(&large_bits, 1_000_000);

    group.bench_function("rank1_1M", |b| {
        b.iter(|| {
            for i in (0..1_000_000).step_by(64) {
                black_box(large_bv.rank1(i));
            }
        })
    });

    group.bench_function("select1_1M", |b| {
        b.iter(|| {
            for k in (0..500_000).step_by(64) {
                black_box(large_bv.select1(k));
            }
        })
    });

    group.bench_function("ones_iter_1M", |b| {
        b.iter(|| {
            black_box(large_bv.ones().count());
        })
    });
    group.finish();
}

fn bench_elias_fano(c: &mut Criterion) {
    let mut group = c.benchmark_group("elias_fano");

    // 10K sorted values in [0, 1M)
    let values: Vec<u32> = (0..10_000).map(|i| i * 100).collect();
    let ef = EliasFano::new(&values, 1_000_001);

    group.bench_function("get_10K", |b| {
        b.iter(|| {
            for i in (0..10_000).step_by(4) {
                black_box(ef.get(i).unwrap());
            }
        })
    });

    group.bench_function("successor_10K", |b| {
        b.iter(|| {
            for target in (0..1_000_000).step_by(400) {
                black_box(ef.successor(target as u32));
            }
        })
    });

    group.bench_function("predecessor_10K", |b| {
        b.iter(|| {
            for target in (0..1_000_000).step_by(400) {
                black_box(ef.predecessor(target as u32));
            }
        })
    });

    group.bench_function("iter_10K", |b| {
        b.iter(|| {
            black_box(ef.iter().count());
        })
    });

    // Serialization roundtrip
    let bytes = ef.to_bytes();
    group.bench_function("from_bytes_10K", |b| {
        b.iter(|| {
            black_box(EliasFano::from_bytes(&bytes).unwrap());
        })
    });
    group.finish();
}

fn bench_partitioned_elias_fano(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioned_ef");

    let values: Vec<u32> = (0..10_000).map(|i| i * 100).collect();
    let pef = PartitionedEliasFano::new(&values, 1_000_001, 128);

    group.bench_function("get_10K", |b| {
        b.iter(|| {
            for i in (0..10_000).step_by(4) {
                black_box(pef.get(i).unwrap());
            }
        })
    });

    group.bench_function("iter_10K", |b| {
        b.iter(|| {
            black_box(pef.iter().count());
        })
    });
    group.finish();
}

fn bench_wavelet_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet_tree");

    // 10K symbols over alphabet of size 256
    let data: Vec<u32> = (0..10_000).map(|i| (i * 7 + 13) % 256).collect();
    let wt = WaveletTree::new(&data, 256);

    group.bench_function("access_10K", |b| {
        b.iter(|| {
            for i in (0..10_000).step_by(4) {
                black_box(wt.access(i));
            }
        })
    });

    group.bench_function("rank_10K", |b| {
        b.iter(|| {
            for i in (0..10_000).step_by(16) {
                black_box(wt.rank(42, i));
            }
        })
    });

    group.bench_function("select_10K", |b| {
        let total = wt.rank(42, 10_000);
        b.iter(|| {
            for k in 0..total {
                black_box(wt.select(42, k));
            }
        })
    });

    // Serialization roundtrip
    let bytes = wt.to_bytes();
    group.bench_function("from_bytes_10K", |b| {
        b.iter(|| {
            black_box(WaveletTree::from_bytes(&bytes).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_bitvector,
    bench_elias_fano,
    bench_partitioned_elias_fano,
    bench_wavelet_tree,
);
criterion_main!(benches);
