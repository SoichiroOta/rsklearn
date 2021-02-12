extern crate ndarray;
use ndarray::prelude::*;
use ndarray::{Array, Ix1, Ix2, Axis};
use ndarray_stats::QuantileExt;

pub struct Linear {
    epochs: usize,
    lr: f64,
    early_stop: Option<f64>,
    beta: Option<Array::<f64, Ix1>>,
    norm: Option<Array::<f64, Ix2>>,
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

    pub fn fit_norm(&mut self, x: Array::<f64, Ix2>, y: Array::<f64, Ix1>) -> &Self {
        let mut norm = Array::<f64, Ix2>::zeros((x.shape()[1] + 1, 2));

        norm.slice_mut(s![0, 0]).fill(*y.min().unwrap());
        norm.slice_mut(s![0, 1]).fill(*y.max().unwrap());

        let x_min = x.map_axis(Axis(0), |view| view.iter().fold(0.0/0.0, |m, v| v.min(m)));
        norm.slice_mut(s![1.., 0]).assign(&x_min);
        let x_max = x.map_axis(Axis(0), |view| view.iter().fold(0.0/0.0, |m, v| v.max(m)));
        norm.slice_mut(s![1.., 1]).assign(&x_max);

        self.norm = Some(norm);
        self
    }

    pub fn normalize(&self, x: Array::<f64, Ix2>, y: Option<Array::<f64, Ix1>>) -> (Array::<f64, Ix2>, Option<Array::<f64, Ix1>>) {
        let norm = match &self.norm {
            &None => {None},
            _ => {self.norm.as_ref()}
        }.unwrap();
        let mut l = &norm.slice(s![1.., 1]) - &norm.slice(s![1.., 0]);
        l = l.mapv(|v: f64| if v == 0.0 {
            1.0
        } else {
            v
        });
        let mut p = x - norm.slice(s![1.., 0]);
        p = &p / &l;
        let p_q = match y {
            None => {
                (p, None)
            },
            Some(y) => {
                let mut q = y;
                if !(&norm[[0, 1]] == &norm[[0, 0]]) {
                    q = q.mapv(|v: f64| v - &norm[[0, 0]]);
                    q = q / (norm[[0, 1]] - norm[[0, 0]]);
                }
                (p, Some(q))
            }
        };
        p_q
    }

    pub fn r2(&self, y: Array::<f64, Ix1>, z: Array::<f64, Ix1>) -> f64 {
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

    pub fn set_beta(&mut self, beta: Array::<f64, Ix1>) -> &Self {
        self.beta = Some(beta);
        self
    }

    pub fn fit(&mut self, x: &Array::<f64, Ix2>, y: &Array::<f64, Ix1>) -> &Self {
        self.fit_norm(x.clone(), y.clone());
        let (x_, y_) = self.normalize(x.clone(), Some(y.clone()));

        let mut beta = Array::<f64, Ix1>::zeros(x_.shape()[1] + 1);

        let data_len = x_.shape()[0];
        let dim = x_.shape()[1];
        for _ in 0..self.epochs {
            for _ in 0..data_len {
                let p = x_.to_owned().index_axis(Axis(0), 0).to_owned();
                self.beta = Some(beta.clone());
                let z = &self.predict(p.clone().into_shape((1, dim)).unwrap(), true);
                let err = (z.clone()[[0]] - y_.as_ref().unwrap().clone()[[0]]) * self.lr;
                let minus_delta = p.clone().mapv(|v: f64| -(v * err));

                beta[[0]] -= err;
                beta.slice_mut(s![1..]).assign(&minus_delta);
            }
            if !self.early_stop.is_none() {
                let z = self.predict(x.clone(), true);
                let s = self.r2(y.clone(), z.clone());
                if self.early_stop.unwrap() <= s {
                    break;
                }
            }
        }
        self
    }

    pub fn predict(&self, x: Array::<f64, Ix2>, normalized: bool) -> Array::<f64, Ix1> {
        let prediction = if normalized == false {
            let (x_, _) = &self.normalize(x, None);

            let beta = &self.beta.as_ref().unwrap();
            let mut z = Array::<f64, Ix1>::zeros(x_.shape()[0]);
            z.fill(beta[[0]]);
            
            let beta_len = beta.shape()[0];
            let coef_len = beta_len - 1;
            let mut coef = Array::<f64, Ix1>::zeros(coef_len);
            for i in 0..coef_len {
                coef[[i]] = beta[[i+1]];
            }
            z = &z + &x_.dot(&coef);
            let norm = &self.norm.as_ref().unwrap();
            z.mapv(|v: f64| v * (norm[[0, 1]] - norm[[0, 0]]) * norm[[0, 0]]);
            z
        } else {
            let beta = &self.beta.as_ref().unwrap();
            let mut z = Array::<f64, Ix1>::zeros(x.shape()[0]);
            z.fill(beta[[0]]);
            
            let beta_len = beta.shape()[0];
            let coef_len = beta_len - 1;
            let mut coef = Array::<f64, Ix1>::zeros(coef_len);
            for i in 0..coef_len {
                coef[[i]] = beta[[i+1]];
            }
            z = &z + &x.dot(&coef);
            z
        };
        prediction
    }

    pub fn str(&self) -> String {
        match self.beta {
            None => String::from("0.0"),
            _ => {
                let mut s = Vec::new();
                let beta = self.beta.as_ref().unwrap();
                s.push(beta[[0]].to_string());
                for i in 1..self.beta.as_ref().unwrap().shape()[0] {
                    s.push(format!(" + feat[ {} ] * {}", i.to_string(), beta[[i]].to_string()));
                }
                String::from(s.iter().cloned().collect::<String>())
            }
        }      
    }
}
