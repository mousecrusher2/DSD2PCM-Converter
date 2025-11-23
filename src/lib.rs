use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod fir_decimator {
    use cblas::ddot;
    use numpy::{
        IntoPyArray as _, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
        ndarray::{Array2, Axis},
    };
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn fir_decimate_chunk_core<'py>(
        py: Python<'py>,
        dsd_ext: PyReadonlyArray2<f32>,
        taps: PyReadonlyArray1<f32>,
        decim: usize,
        phase_init: usize,
        over_lap: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        let dsd_ext = dsd_ext.as_array();
        let dsd_ext = dsd_ext
            .axis_iter(Axis(1))
            .map(|x| x.iter().map(|&y| y as f64).collect::<Vec<f64>>())
            .collect::<Vec<_>>();
        let taps = taps
            .as_slice()
            .unwrap()
            .iter()
            .rev()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();
        let res = py
            .detach(|| fir_decimate_chunk_core_inner(&dsd_ext, &taps, decim, phase_init, over_lap));
        res.into_pyarray(py)
    }

    fn fir_decimate_chunk_core_inner(
        dsd_ext: &[Vec<f64>],
        taps: &[f64],
        decim: usize,
        phase_init: usize,
        over_lap: usize,
    ) -> Array2<f64> {
        let n_ext = dsd_ext.len();
        let ch = dsd_ext.len();
        let l = taps.len();

        let main_len = dsd_ext[0].len().saturating_sub(over_lap);
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
        for oi in 0..n_out {
            let i = first_i + oi * decim;
            for c in 0..ch {
                let x = &dsd_ext[c][i - (l - 1)..=i];
                let y = &taps;
                out[[oi, c]] = dot(x, y);
            }
        }
        out
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        unsafe { ddot(a.len() as i32, a, 1, b, 1) }
    }
}
