//! SIMD implementations of distance functions.

pub(crate) fn distance_simd_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_castpd256_pd128, _mm256_extractf128_pd, _mm256_loadu_pd,
            _mm256_mul_pd, _mm256_setzero_pd, _mm256_sub_pd, _mm_add_pd, _mm_add_sd, _mm_cvtsd_f64,
            _mm_unpackhi_pd,
        };
        debug_assert_eq!(lhs.len(), rhs.len());

        unsafe {
            let mut acc_4x = _mm256_setzero_pd();
            for (lh_slice, rh_slice) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
                let lh_4x = _mm256_loadu_pd(lh_slice.as_ptr());
                let rh_4x = _mm256_loadu_pd(rh_slice.as_ptr());
                let diff = _mm256_sub_pd(lh_4x, rh_4x);
                let diff_squared = _mm256_mul_pd(diff, diff);
                acc_4x = _mm256_add_pd(diff_squared, acc_4x);
            }

            // Sum up the components in `acc_4x`
            let acc_high = _mm256_extractf128_pd(acc_4x, 1);
            let acc_low = _mm256_castpd256_pd128(acc_4x);
            let acc_2x = _mm_add_pd(acc_high, acc_low);

            let mut acc = _mm_add_pd(acc_2x, _mm_unpackhi_pd(acc_2x, acc_2x));
            acc = _mm_add_sd(acc, _mm_unpackhi_pd(acc, acc));

            let remaining_elements = &lhs[lhs.len() - lhs.len() % 4..];
            let mut residual = 0.0;
            for (&lh, &rh) in remaining_elements
                .iter()
                .zip(rhs[lhs.len() - lhs.len() % 4..].iter())
            {
                residual += (lh - rh).powi(2);
            }

            let residual = residual + _mm_cvtsd_f64(acc);
            residual.sqrt()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (*a - *b).pow(2))
        .sum::<f32>()
        .sqrt()
}

pub(crate) fn distance_simd_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
            _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
            _mm_movehl_ps, _mm_shuffle_ps,
        };
        debug_assert_eq!(lhs.len(), rhs.len());

        unsafe {
            let mut acc_8x = _mm256_setzero_ps();
            for (lh_slice, rh_slice) in lhs.chunks_exact(8).zip(rhs.chunks_exact(8)) {
                let lh_8x = _mm256_loadu_ps(lh_slice.as_ptr());
                let rh_8x = _mm256_loadu_ps(rh_slice.as_ptr());
                let diff = _mm256_sub_ps(lh_8x, rh_8x);
                let diff_squared = _mm256_mul_ps(diff, diff);
                acc_8x = _mm256_add_ps(diff_squared, acc_8x);
            }

            // Sum up the components in `acc_8x`
            let acc_high = _mm256_extractf128_ps(acc_8x, 1);
            let acc_low = _mm256_castps256_ps128(acc_8x);
            let acc_4x = _mm_add_ps(acc_high, acc_low);

            let mut acc = _mm_add_ps(acc_4x, _mm_movehl_ps(acc_4x, acc_4x));
            acc = _mm_add_ss(acc, _mm_shuffle_ps(acc, acc, 0x55));

            let remaining_elements = &lhs[lhs.len() - lhs.len() % 8..];
            let mut residual = 0.0;
            for (&lh, &rh) in remaining_elements
                .iter()
                .zip(rhs[lhs.len() - lhs.len() % 8..].iter())
            {
                residual += (lh - rh).powi(2);
            }

            let residual = residual + _mm_cvtss_f32(acc);
            residual.sqrt()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (*a - *b).pow(2) as f32)
        .sum::<f32>()
        .sqrt()
}
