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

fn bench_bitvector_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("bv_density");

    // 10M bits at various densities
    let n = 10_000_000usize;
    let num_words = n / 64;

    // Sparse (1%)
    let sparse: Vec<u64> = (0..num_words)
        .map(|i| if i % 100 == 0 { 1u64 } else { 0 })
        .collect();
    let bv_sparse = BitVector::new(&sparse, n);
    let sparse_ones = bv_sparse.rank1(n);

    group.bench_function("rank1_sparse_10M", |b| {
        b.iter(|| {
            for i in (0..n).step_by(1000) {
                black_box(bv_sparse.rank1(i));
            }
        })
    });

    group.bench_function("select1_sparse_10M", |b| {
        b.iter(|| {
            for k in (0..sparse_ones).step_by(100) {
                black_box(bv_sparse.select1(k));
            }
        })
    });

    // Dense (99%)
    let dense: Vec<u64> = (0..num_words)
        .map(|i| if i % 100 == 0 { u64::MAX - 1 } else { u64::MAX })
        .collect();
    let bv_dense = BitVector::new(&dense, n);
    let dense_ones = bv_dense.rank1(n);

    group.bench_function("rank1_dense_10M", |b| {
        b.iter(|| {
            for i in (0..n).step_by(1000) {
                black_box(bv_dense.rank1(i));
            }
        })
    });

    group.bench_function("select1_dense_10M", |b| {
        b.iter(|| {
            for k in (0..dense_ones).step_by(1000) {
                black_box(bv_dense.select1(k));
            }
        })
    });

    // Uniform 50%
    let uniform: Vec<u64> = vec![0xAAAAAAAAAAAAAAAAu64; num_words];
    let bv_uniform = BitVector::new(&uniform, n);

    group.bench_function("rank1_uniform_10M", |b| {
        b.iter(|| {
            for i in (0..n).step_by(1000) {
                black_box(bv_uniform.rank1(i));
            }
        })
    });

    group.bench_function("select1_uniform_10M", |b| {
        b.iter(|| {
            for k in (0..5_000_000).step_by(1000) {
                black_box(bv_uniform.select1(k));
            }
        })
    });

    group.finish();
}

fn bench_ef_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("ef_scale");

    // 100K elements
    let values_100k: Vec<u32> = (0..100_000).map(|i| i * 40).collect();
    let ef_100k = EliasFano::new(&values_100k, 4_000_001);

    group.bench_function("successor_100K", |b| {
        b.iter(|| {
            for target in (0..4_000_000u32).step_by(4000) {
                black_box(ef_100k.successor(target));
            }
        })
    });

    group.bench_function("iter_100K", |b| {
        b.iter(|| {
            black_box(ef_100k.iter().count());
        })
    });

    // Clustered data (simulates posting list)
    let mut clustered = Vec::new();
    for cluster in 0..100 {
        let base = cluster * 100_000;
        for j in 0..1000 {
            clustered.push(base + j * 3);
        }
    }
    let ef_clustered = EliasFano::new(&clustered, 10_000_001);
    let pef_clustered = PartitionedEliasFano::new(&clustered, 10_000_001, 128);

    group.bench_function("ef_successor_clustered_100K", |b| {
        b.iter(|| {
            for target in (0..10_000_000u32).step_by(10_000) {
                black_box(ef_clustered.successor(target));
            }
        })
    });

    group.bench_function("pef_get_clustered_100K", |b| {
        b.iter(|| {
            for i in (0..100_000).step_by(4) {
                black_box(pef_clustered.get(i).unwrap());
            }
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
    bench_bitvector_density,
    bench_ef_scale,
);
criterion_main!(benches);
