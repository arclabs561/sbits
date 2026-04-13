#![no_main]
use libfuzzer_sys::fuzz_target;
use sbits::elias_fano::EliasFano;

fuzz_target!(|data: &[u8]| {
    // Fuzz the deserializer with arbitrary bytes -- should never panic.
    let _ = EliasFano::from_bytes(data);
});
