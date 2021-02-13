use ndarray::prelude::*;
use ndarray::{Array, Axis, Ix1, Ix2};
use ndarray_stats::QuantileExt;

pub struct Linear {
    epochs: usize,
    lr: f64,
    early_stop: Option<f64>,
    beta: Option<f64>,
    norm: Option<Array<f64, Ix2>>,
}

/// Configure and fit a linear regression model
impl Linear {
    /// Create a default linear regression model.
    pub fn new(epochs: usize, lr: f64, early_stop: Option<f64>) -> Linear {
        Linear {
            epochs,
            lr,
            early_stop,
            beta: None,
            norm: None,
        }
    }

    pub fn fit_norm(mut self, x: Array<f64, Ix2>, y: Array<f64, Ix1>) -> Self {
        let mut norm = Array::<f64, Ix2>::zeros((x.shape()[1] + 1, 2));

        norm.slice_mut(s![0, 0]).fill(*y.min().unwrap());
        norm.slice_mut(s![0, 1]).fill(*y.max().unwrap());

        let x_min = x.map_axis(Axis(0), |view| view.iter().fold(0.0 / 0.0, |m, v| v.min(m)));
        norm.slice_mut(s![1.., 0]).assign(&x_min);
        let x_max = x.map_axis(Axis(0), |view| view.iter().fold(0.0 / 0.0, |m, v| v.max(m)));
        norm.slice_mut(s![1.., 1]).assign(&x_max);

        self.norm = Some(norm);
        self
    }

    pub fn normalize(
        self,
        x: Array<f64, Ix2>,
        y: Option<Array<f64, Ix1>>,
    ) -> (Array<f64, Ix2>, Option<Array<f64, Ix1>>) {
        let norm = self.norm.unwrap();
        let mut l = &norm.slice(s![1.., 1]) - &norm.slice(s![1.., 0]);
        l = l.mapv(|v: f64| if v == 0.0 { 1.0 } else { v });
        let mut p = &x - &norm.slice(s![1.., 0]);
        p = &p / &l;
        let p_q = match y {
            None => (p, y),
            Some(y) => {
                let mut q = y;
                if !(norm[[0, 1]] == norm[[0, 0]]) {
                    q = q.mapv(|v: f64| v - norm[[0, 0]]);
                    q = &q / (&norm[[0, 1]] - &norm[[0, 0]])
                }
                (p, Some(q))
            }
        };
        p_q
    }

    pub fn r2(self, y: Array<f64, Ix1>, z: Array<f64, Ix1>) -> f64 {
        let mut y_minus_z_pow2 = &y - &z;
        y_minus_z_pow2 = y_minus_z_pow2.mapv(|v: f64| v.powi(2));
        let mn = y_minus_z_pow2.sum();

        let y_mean = y.mean();
        let y_minus_y_mean_pow2 = y.mapv(|v: f64| v - y_mean.unwrap());
        let dn = y_minus_y_mean_pow2.sum();

        let r2 = if dn == 0.0 {
            f64::INFINITY
        } else {
            1.0 - mn / dn
        };
        r2
    }
}

impl Default for Linear {
    fn default() -> Self {
        Linear {
            epochs: 20,
            lr: 0.01,
            early_stop: None,
            beta: None,
            norm: None,
        }
    }
}
