extern crate ndarray;
use ndarray::{Array, Ix1, Ix2};
use ndarray::prelude::*;

use rsklearn::entropy::{gini};
use rsklearn_zeror::{ZeroRule};
use rsklearn_linear::{Linear};

pub struct DecisionStump {
    metric: fn(Array<f64, Ix2>) -> f64,
    feat_index: usize,
    feat_val: f64,
    score: f64,
}

impl Default for DecisionStump {
    fn default() -> Self {
        DecisionStump {
            metric: gini,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
        }
    }
}

impl DecisionStump {
    pub fn new(metric: fn(Array<f64, Ix2>) -> f64) -> DecisionStump {
        DecisionStump {
            metric: metric,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
        }
    }

    pub fn make_split(&self, feat: Array::<f64, Ix1>, val: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, v) in feat.iter().enumerate() {
            if v < &val {
                left.push(i)
            } else {
                right.push(i)
            }
        }
        (left, right)
    }

    pub fn make_loss(&self, y1: Array::<f64, Ix2>, y2: Array::<f64, Ix2>) -> f64 {
        if y1.shape()[0] == 0 || y2.shape()[0] == 0 {
            return f64::INFINITY
        }
        let total = y1.shape()[0] as f64 + y2.shape()[0] as f64;
        let metric = self.metric;
        let m1 = metric(y1.clone()) * (y1.shape()[0] as f64 / total);
        let m2 = metric(y2.clone()) * (y2.shape()[0] as f64 / total);
        m1 + m2
    }

    pub fn split_tree(&mut self, x: Array::<f64, Ix2>, y: Array::<f64, Ix2>) -> (Vec<usize>, Vec<usize>) {
        self.feat_index = 0;
        self.feat_val = f64::INFINITY;
        let mut score = f64::INFINITY;
        let mut left = (0..x.shape()[0]).collect::<Vec<usize>>();
        let mut right = Vec::new();
        for i in 0..x.shape()[1] {
            let feat = x.slice(s![.., i]);
            for val in feat.iter() {
                let (l, r) = self.make_split(feat.to_owned(), *val);
                let mut y_l = Array::<f64, Ix2>::zeros((y.shape()[0], l.len()));
                for l_elm in l.iter() {
                    y_l.slice_mut(s![*l_elm, ..]).assign(&y.slice(s![*l_elm, ..]))
                }
                let mut y_r = Array::<f64, Ix2>::zeros((y.shape()[0], r.len()));
                for r_elm in r.iter() {
                    y_r.slice_mut(s![*r_elm, ..]).assign(&y.slice(s![*r_elm, ..]))
                }
                let loss = self.make_loss(y_l, y_r);
                if score > loss {
                    score = loss;
                    left = l;
                    right = r;
                    self.feat_index = i;
                    self.feat_val = *val;
                }
            }
        }
        self.score = score;
        (left, right)
    }
}
