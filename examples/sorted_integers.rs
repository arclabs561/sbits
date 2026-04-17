/// Store sorted integers in compressed form with fast lookup.
///
/// Elias-Fano encoding stores a sorted sequence of integers using
/// close to the information-theoretic minimum space, while still
/// supporting O(1) access-by-index and fast successor queries.
///
/// Typical use: posting lists (sorted doc IDs matching a search term),
/// compressed indexes, or any monotone integer sequence.
///
/// ```sh
/// cargo run --example sorted_integers
/// ```
use sbits::elias_fano::EliasFano;

fn main() {
    // Simulate a posting list: 1000 doc IDs out of a 1M-document corpus.
    let mut values: Vec<u64> = (0..1_000_000u64)
        .step_by(1000)
        .collect();
    // Add some irregular gaps to make it realistic.
    values.push(42);
    values.push(7777);
    values.push(999_999);
    values.sort();
    values.dedup();

    let universe = 1_000_000;
    let ef = EliasFano::new(&values, universe);

    println!(
        "Stored {} sorted values (universe 0..{})\n",
        ef.len(),
        ef.universe_size()
    );

    // Random access.
    println!("First 5 values:");
    for i in 0..5 {
        println!("  ef[{}] = {}", i, ef.get(i).unwrap());
    }

    // Successor: find the first doc ID >= target.
    println!("\nSuccessor queries:");
    for target in [43, 5000, 999_998] {
        println!("  successor({}) = {:?}", target, ef.successor(target));
    }

    // Space comparison.
    let raw_bytes = values.len() * std::mem::size_of::<u64>();
    let ef_bytes = ef.heap_bytes();
    let ratio = raw_bytes as f64 / ef_bytes as f64;
    println!(
        "\nSpace: {} bytes compressed vs {} bytes raw ({:.1}x smaller)",
        ef_bytes, raw_bytes, ratio,
    );
}
