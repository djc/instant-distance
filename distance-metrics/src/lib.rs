#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod simd_sse;

#[cfg(target_arch = "x86_64")]
pub mod simd_avx;

#[cfg(target_arch = "aarch64")]
pub mod simd_neon;

/// Defines how to compare vectors
pub trait Metric {
    /// Greater the value - more distant the vectors
    fn distance(v1: &[f32], v2: &[f32]) -> f32;
}

#[cfg(target_arch = "x86_64")]
const MIN_DIM_SIZE_AVX: usize = 32;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
const MIN_DIM_SIZE_SIMD: usize = 16;

#[derive(Clone, Copy)]
pub struct EuclidMetric {}

impl Metric for EuclidMetric {
    fn distance(v1: &[f32], v2: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && v1.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { simd_avx::euclid_distance_avx(v1, v2) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { simd_sse::euclid_distance_sse(v1, v2) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { simple_neon::euclid_distance_neon(v1, v2) };
            }
        }

        euclid_distance(v1, v2)
    }
}

pub fn euclid_distance(v1: &[f32], v2: &[f32]) -> f32 {
    let s: f32 = v1
        .iter()
        .copied()
        .zip(v2.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    s.abs().sqrt()
}

pub fn legacy_distance(lhs: &FloatArray, rhs: &FloatArray) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_load_ps,
            _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_fmadd_ps,
            _mm_load_ps, _mm_movehl_ps, _mm_shuffle_ps, _mm_sub_ps,
        };
        debug_assert_eq!(lhs.0.len() % 8, 4);

        unsafe {
            let mut acc_8x = _mm256_setzero_ps();
            for (lh_slice, rh_slice) in lhs.0.chunks_exact(8).zip(rhs.0.chunks_exact(8)) {
                let lh_8x = _mm256_load_ps(lh_slice.as_ptr());
                let rh_8x = _mm256_load_ps(rh_slice.as_ptr());
                let diff = _mm256_sub_ps(lh_8x, rh_8x);
                acc_8x = _mm256_fmadd_ps(diff, diff, acc_8x);
            }

            let mut acc_4x = _mm256_extractf128_ps(acc_8x, 1); // upper half
            let right = _mm256_castps256_ps128(acc_8x); // lower half
            acc_4x = _mm_add_ps(acc_4x, right); // sum halves

            let lh_4x = _mm_load_ps(lhs.0[DIMENSIONS - 4..].as_ptr());
            let rh_4x = _mm_load_ps(rhs.0[DIMENSIONS - 4..].as_ptr());
            let diff = _mm_sub_ps(lh_4x, rh_4x);
            acc_4x = _mm_fmadd_ps(diff, diff, acc_4x);

            let lower = _mm_movehl_ps(acc_4x, acc_4x);
            acc_4x = _mm_add_ps(acc_4x, lower);
            let upper = _mm_shuffle_ps(acc_4x, acc_4x, 0x1);
            acc_4x = _mm_add_ss(acc_4x, upper);
            _mm_cvtss_f32(acc_4x)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    lhs.0
        .iter()
        .zip(rhs.0.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
}

#[repr(align(32))]
pub struct FloatArray(pub [f32; DIMENSIONS]);

const DIMENSIONS: usize = 300;
