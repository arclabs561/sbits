use proptest::prelude::*;
use sbits::bitvec::BitVector;

proptest! {
    #[test]
    fn test_bitvector_rank_property(
        bits in prop::collection::vec(any::<u64>(), 1..100),
        len_mult in 0..64usize,
    ) {
        let len = (bits.len() * 64).saturating_sub(len_mult);
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

        prop_assert_eq!(bv.rank1(len), expected_total);

        // Check individual ranks at random points
        for i in (0..len).step_by(13) {
            let mut expected_rank = 0;
            for j in 0..i {
                let word_idx = j / 64;
                let bit_idx = j % 64;
                if (bits[word_idx] & (1 << bit_idx)) != 0 {
                    expected_rank += 1;
                }
            }
            prop_assert_eq!(bv.rank1(i), expected_rank);
            prop_assert_eq!(bv.rank0(i), i - expected_rank);
        }

        // Check select1 for every set bit
        let mut count = 0;
        for i in 0..len {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if (bits[word_idx] & (1 << bit_idx)) != 0 {
                prop_assert_eq!(bv.select1(count), Some(i));
                count += 1;
            }
        }
        prop_assert_eq!(bv.select1(count), None);

        // Check select0 for every unset bit
        let mut count0 = 0;
        for i in 0..len {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if (bits[word_idx] & (1 << bit_idx)) == 0 {
                prop_assert_eq!(bv.select0(count0), Some(i));
                count0 += 1;
            }
        }
        prop_assert_eq!(bv.select0(count0), None);
    }
}

use sbits::elias_fano::EliasFano;
use sbits::wavelet::WaveletTree;

proptest! {
    #[test]
    fn test_elias_fano_property(
        mut values in prop::collection::vec(0..10000u32, 1..100),
    ) {
        values.sort();
        values.dedup();
        if values.is_empty() { return Ok(()); }

        let universe_size = values.last().copied().unwrap() + 100;
        let ef = EliasFano::new(&values, universe_size);

        prop_assert_eq!(ef.len(), values.len());

        for (i, &expected) in values.iter().enumerate() {
            prop_assert_eq!(ef.get(i).unwrap(), expected);
        }
    }

    #[test]
    fn test_wavelet_tree_property(
        input in prop::collection::vec(0..100u32, 1..100),
    ) {
        let sigma = input.iter().max().copied().unwrap_or(0) + 1;
        let wt = WaveletTree::new(&input, sigma);

        prop_assert_eq!(wt.len(), input.len());

        for (i, &expected) in input.iter().enumerate() {
            prop_assert_eq!(wt.access(i), expected);
        }

        // Check rank for each symbol
        for symbol in 0..sigma {
            let mut expected_rank = 0;
            for (i, &v) in input.iter().enumerate() {
                prop_assert_eq!(wt.rank(symbol, i), expected_rank);
                if v == symbol {
                    expected_rank += 1;
                }
            }
            prop_assert_eq!(wt.rank(symbol, input.len()), expected_rank);
        }
    }
}
