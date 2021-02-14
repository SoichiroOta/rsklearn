extern crate ndarray;
use ndarray::prelude::*;
use ndarray::{Array, Ix1, Ix2};

use rsklearn::entropy::gini;
use rsklearn_linear::Linear;
use rsklearn_zeror::{zero_rule, ZeroRule};

pub struct DecisionStump<T> {
    metric: fn(Array<f64, Ix2>) -> f64,
    leaf: fn() -> T,
    left: Option<T>,
    right: Option<T>,
    feat_index: usize,
    feat_val: f64,
    score: f64,
}

impl Default for DecisionStump<ZeroRule> {
    fn default() -> Self {
        DecisionStump {
            metric: gini,
            leaf: zero_rule,
            left: None,
            right: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
        }
    }
}

impl DecisionStump<ZeroRule> {
    pub fn new(metric: fn(Array<f64, Ix2>) -> f64) -> DecisionStump<ZeroRule> {
        DecisionStump {
            metric: metric,
            leaf: zero_rule,
            left: None,
            right: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
        }
    }

    pub fn make_split(&self, feat: Array<f64, Ix1>, val: f64) -> (Vec<usize>, Vec<usize>) {
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

    pub fn make_loss(&self, y1: Array<f64, Ix2>, y2: Array<f64, Ix2>) -> f64 {
        if y1.shape()[0] == 0 || y2.shape()[0] == 0 {
            return f64::INFINITY;
        }
        let total = y1.shape()[0] as f64 + y2.shape()[0] as f64;
        let metric = self.metric;
        let m1 = metric(y1.clone()) * (y1.shape()[0] as f64 / total);
        let m2 = metric(y2.clone()) * (y2.shape()[0] as f64 / total);
        m1 + m2
    }

    pub fn split_tree(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix2>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.feat_index = 0;
        self.feat_val = f64::INFINITY;
        let mut score = f64::INFINITY;
        let mut left = (0..x.shape()[0]).collect::<Vec<usize>>();
        let mut right = Vec::new();
        for i in 0..x.shape()[1] {
            let feat = x.slice(s![.., i]);
            for val in feat.iter() {
                let (l, r) = self.make_split(feat.to_owned(), *val);
                let mut y_l = Array::<f64, Ix2>::zeros((l.len(), y.shape()[1]));
                for l_elm in l.iter() {
                    y_l.slice_mut(s![*l_elm, ..])
                        .assign(&y.slice(s![*l_elm, ..]));
                }
                let mut y_r = Array::<f64, Ix2>::zeros((r.len(), y.shape()[1]));
                for r_elm in r.iter() {
                    y_r.slice_mut(s![*r_elm, ..])
                        .assign(&y.slice(s![*r_elm, ..]));
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

    pub fn fit(&mut self, x: Array<f64, Ix2>, y: Array<f64, Ix2>) -> &Self {
        let leaf = self.leaf;
        let (left, right) = self.split_tree(x.clone(), y.clone());
        if left.len() > 0 {
            let mut x_left = Array::<f64, Ix2>::zeros((left.len(), x.clone().shape()[1]));
            for left_elm in left.iter() {
                x_left
                    .slice_mut(s![*left_elm, ..])
                    .assign(&x.slice(s![*left_elm, ..]));
            }
            let mut y_left = Array::<f64, Ix2>::zeros((left.len(), y.clone().shape()[1]));
            for left_elm in left.iter() {
                y_left
                    .slice_mut(s![*left_elm, ..])
                    .assign(&y.slice(s![*left_elm, ..]));
            }
            self.left = Some(leaf().fit(x_left, y_left));
        }
        if right.len() > 0 {
            let mut x_right = Array::<f64, Ix2>::zeros((right.len(), x.shape()[1]));
            for right_elm in right.iter() {
                x_right
                    .slice_mut(s![*right_elm, ..])
                    .assign(&x.slice(s![*right_elm, ..]));
            }
            let mut y_right = Array::<f64, Ix2>::zeros((right.len(), y.shape()[1]));
            for right_elm in right.iter() {
                y_right
                    .slice_mut(s![*right_elm, ..])
                    .assign(&y.slice(s![*right_elm, ..]));
            }
            self.right = Some(leaf().fit(x_right, y_right));
        }
        self
    }

    pub fn predict(&self, x: Array<f64, Ix2>) -> Option<Array<f64, Ix2>> {
        let feat = x.slice(s![.., self.feat_index]);
        let val = self.feat_val;
        let (l, r) = self.make_split(feat.to_owned(), val.to_owned());
        if l.len() > 0 && r.len() > 0 {
            let mut x_l = Array::<f64, Ix2>::zeros((l.len(), x.shape()[1]));
            for l_elm in l.iter() {
                x_l.slice_mut(s![*l_elm, ..])
                    .assign(&x.slice(s![*l_elm, ..]))
            }
            let left = self.left.as_ref().unwrap().predict(x_l);
            let mut x_r = Array::<f64, Ix2>::zeros((r.len(), x.shape()[1]));
            for r_elm in r.iter() {
                x_r.slice_mut(s![*r_elm, ..])
                    .assign(&x.slice(s![*r_elm, ..]))
            }
            let right = self.right.as_ref().unwrap().predict(x_r);
            let mut z = Array::<f64, Ix2>::zeros((x.shape()[0], left.shape()[1]));
            for l_elm in l.iter() {
                z.slice_mut(s![*l_elm, ..])
                    .assign(&left.slice(s![*l_elm, ..]));
            }
            for r_elm in r.iter() {
                z.slice_mut(s![*r_elm, ..])
                    .assign(&right.slice(s![*r_elm, ..]));
            }
            Some(z)
        } else if l.len() > 0 {
            Some(self.left.as_ref().unwrap().predict(x))
        } else if r.len() > 0 {
            Some(self.right.as_ref().unwrap().predict(x))
        } else {
            None
        }
    }
}
