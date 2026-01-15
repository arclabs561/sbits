#![no_main]
use libfuzzer_sys::fuzz_target;
use sbits::bitvec::BitVector;

fuzz_target!(|data: (Vec<u64>, usize)| {
    let (bits, len_raw) = data;
    if bits.is_empty() {
        return;
    }

    let len = len_raw % (bits.len() * 64);
    if len == 0 {
        return;
    }

    let bv = BitVector::new(&bits, len);

    // Check total rank
    let mut expected_total = 0;
    for i in 0..len {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        if (bits[word_idx] & (1 << bit_idx)) != 0 {
            expected_total += 1;
        }
    }

    assert_eq!(bv.rank1(len), expected_total);

    // Check select1 for a random rank
    if expected_total > 0 {
        let k = (len_raw / 13) % expected_total;
        if let Some(pos) = bv.select1(k) {
            assert!(pos < len);
            assert_eq!(bv.rank1(pos + 1), k + 1);
            assert_eq!(bv.rank1(pos), k);
        } else {
            panic!(
                "select1({}) failed for expected_total={}",
                k, expected_total
            );
        }
    }
});
