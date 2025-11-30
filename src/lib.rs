use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod fir_decimator {
    use std::{arch::x86_64::*, ops::Deref};

    use numpy::{
        IntoPyArray as _, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
        ndarray::{Array2, Axis},
    };
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn fir_decimate_chunk_core<'py>(
        py: Python<'py>,
        dsd_ext: PyReadonlyArray2<u8>,
        taps: PyReadonlyArray1<f64>,
        decim: usize,
        phase_init: usize,
        over_lap: usize,
        main_len: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        let dsd_ext = dsd_ext.as_array();
        let taps = taps.as_slice().unwrap();
        let res = py.detach(|| {
            let dsd_ext = dsd_ext
                .axis_iter(Axis(1))
                .map(|x| {
                    x.iter()
                        .map(|&y| ((y as i64 * 2) - 1) as f64)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>();
            let taps = AArray::new(taps);
            fir_decimate_chunk_core_inner(
                &dsd_ext, &taps, decim, phase_init, over_lap, main_len,
            )
        });
        res.into_pyarray(py)
    }

    fn fir_decimate_chunk_core_inner(
        dsd_ext: &[Vec<f64>],
        taps: &[f64],
        decim: usize,
        phase_init: usize,
        over_lap: usize,
        main_len: usize,
    ) -> Array2<f64> {
        let ch = dsd_ext.len();
        if ch == 0 {
            return Array2::zeros((0, 0));
        }
        let n_ext = dsd_ext[0].len();
        let l = taps.len();

        let main_start = over_lap;
        let main_end = over_lap + main_len;

        if main_len == 0 {
            return Array2::zeros((0, ch));
        }
        if main_end > n_ext {
            panic!("main_end > n_ext");
        }

        let r = (phase_init + main_start) % decim;
        let first_i = if r == 0 {
            main_start
        } else {
            main_start + (decim - r)
        };

        if first_i >= main_end {
            return Array2::zeros((0, ch));
        }

        let last_i = main_end - 1;
        let span = last_i - first_i;
        let n_out = span / decim + 1;

        let mut out = Array2::<f64>::zeros((n_out, ch));
        for c in 0..ch {
            for oi in 0..n_out {
                let i = first_i + oi * decim;
                let x = &dsd_ext[c][i - (l - 1)..=i];
                let y = &taps;
                out[[oi, c]] = ddot(x, y);
            }
        }
        out
    }

    fn ddot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        assert!((bptr as usize).is_multiple_of(32));
        // b is aligned to 32 bytes
        // a is not guaranteed to be aligned
        unsafe {
            let mut sum1 = _mm256_setzero_pd();
            let mut sum2 = _mm256_setzero_pd();
            let mut sum3 = _mm256_setzero_pd();
            let mut sum4 = _mm256_setzero_pd();
            let mut sum5 = _mm256_setzero_pd();
            let mut sum6 = _mm256_setzero_pd();
            let mut sum7 = _mm256_setzero_pd();
            let mut sum8 = _mm256_setzero_pd();
            for i in 0..a.len() / 32 {
                let avec1 = _mm256_loadu_pd(aptr.add(i * 32));
                let bvec1 = _mm256_load_pd(bptr.add(i * 32));
                let avec2 = _mm256_loadu_pd(aptr.add(i * 32 + 4));
                let bvec2 = _mm256_load_pd(bptr.add(i * 32 + 4));
                let avec3 = _mm256_loadu_pd(aptr.add(i * 32 + 8));
                let bvec3 = _mm256_load_pd(bptr.add(i * 32 + 8));
                let avec4 = _mm256_loadu_pd(aptr.add(i * 32 + 12));
                let bvec4 = _mm256_load_pd(bptr.add(i * 32 + 12));
                let avec5 = _mm256_loadu_pd(aptr.add(i * 32 + 16));
                let bvec5 = _mm256_load_pd(bptr.add(i * 32 + 16));
                let avec6 = _mm256_loadu_pd(aptr.add(i * 32 + 20));
                let bvec6 = _mm256_load_pd(bptr.add(i * 32 + 20));
                let avec7 = _mm256_loadu_pd(aptr.add(i * 32 + 24));
                let bvec7 = _mm256_load_pd(bptr.add(i * 32 + 24));
                let avec8 = _mm256_loadu_pd(aptr.add(i * 32 + 28));
                let bvec8 = _mm256_load_pd(bptr.add(i * 32 + 28));
                sum1 = _mm256_fmadd_pd(avec1, bvec1, sum1);
                sum2 = _mm256_fmadd_pd(avec2, bvec2, sum2);
                sum3 = _mm256_fmadd_pd(avec3, bvec3, sum3);
                sum4 = _mm256_fmadd_pd(avec4, bvec4, sum4);
                sum5 = _mm256_fmadd_pd(avec5, bvec5, sum5);
                sum6 = _mm256_fmadd_pd(avec6, bvec6, sum6);
                sum7 = _mm256_fmadd_pd(avec7, bvec7, sum7);
                sum8 = _mm256_fmadd_pd(avec8, bvec8, sum8);
            }
            let sum = _mm256_add_pd(
                _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4)),
                _mm256_add_pd(_mm256_add_pd(sum5, sum6), _mm256_add_pd(sum7, sum8)),
            );
            let sum = sum4xd(sum);
            let sum2 = a[a.len() / 32 * 32..]
                .iter()
                .zip(b[a.len() / 32 * 32..].iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();
            sum + sum2
        }
    }

    struct AArray {
        data_ptr: *mut f64,
        len: usize,
    }

    impl AArray {
        fn new(data: &[f64]) -> Self {
            let len = data.len();
            let layout = std::alloc::Layout::from_size_align(std::mem::size_of_val(data), 32)
                .expect("Failed to create layout");
            unsafe {
                let ptr = std::alloc::alloc(layout) as *mut f64;
                if ptr.is_null() {
                    std::alloc::handle_alloc_error(layout);
                }
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
                AArray { data_ptr: ptr, len }
            }
        }
    }

    impl Drop for AArray {
        fn drop(&mut self) {
            let layout =
                std::alloc::Layout::from_size_align(self.len * std::mem::size_of::<f64>(), 32)
                    .expect("Failed to create layout");
            unsafe {
                std::alloc::dealloc(self.data_ptr as *mut u8, layout);
            }
        }
    }

    impl Deref for AArray {
        type Target = [f64];

        fn deref(&self) -> &Self::Target {
            unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
        }
    }

    #[inline(always)]
    fn sum4xd(x: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(x, 1);
            let lo = _mm256_castpd256_pd128(x);
            let sum2 = _mm_add_pd(hi, lo);
            let hi2 = _mm_unpackhi_pd(sum2, sum2);
            let sum1 = _mm_add_sd(sum2, hi2);
            _mm_cvtsd_f64(sum1)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn boxub() {
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let b = AArray::new(&a);
            assert_eq!(&*b, &a[..]);
        }

        #[test]
        fn ddot_intrin_ub() {
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let b = AArray::new(&a);
            let _c = ddot(&b, &b);
        }
    }
}
