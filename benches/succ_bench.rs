use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sbits::bitvec::BitVector;

fn bench_bitvector(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitvector");
    let bits = vec![0xAAAAAAAAAAAAAAAAu64; 1000]; // 6400 bits, 50% density
    let bv = BitVector::new(&bits, 64000);

    group.bench_function("rank1", |b| {
        b.iter(|| {
            for i in 0..64000 {
                black_box(bv.rank1(i));
            }
        })
    });

    group.bench_function("select1", |b| {
        b.iter(|| {
            for k in 0..32000 {
                black_box(bv.select1(k));
            }
        })
    });
}

criterion_group!(benches, bench_bitvector);
criterion_main!(benches);
