use sail::kernel::cubecl_associative_scan;
use sail::kernel::scan::cpu_associative_scan_slice;

#[test]
fn test_associative_scan_kernel_correctness() {
    // Check that associative scan matches sequential recurrent formula:
    // h_t = a_t * h_{t-1} + b_t
    let a = vec![0.8f32, 0.9, 0.7, 0.5, 0.95, 0.85, 0.9, 0.6];
    let b = vec![1.0f32, -0.5, 2.0, 1.5, -1.0, 0.5, -0.2, 0.8];

    let mut expected = vec![0.0f32; a.len()];
    let mut h = 0.0f32;
    for i in 0..a.len() {
        h = a[i] * h + b[i];
        expected[i] = h;
    }

    let actual = cubecl_associative_scan(&a, &b);
    assert_eq!(actual.len(), expected.len());

    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-6,
            "Mismatch at index {i}: actual {}, expected {}",
            actual[i],
            expected[i]
        );
    }

    let mut out_slice = vec![0.0f32; a.len()];
    cpu_associative_scan_slice(&a, &b, &mut out_slice);
    for i in 0..actual.len() {
        assert_eq!(actual[i], out_slice[i]);
    }
}
