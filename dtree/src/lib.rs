extern crate ndarray;
use ndarray::prelude::*;
use ndarray::{Array, Ix1, Ix2};

use rsklearn::entropy::gini;
use rsklearn_dstump::{linear_leaf, zero_rule_leaf, DecisionStump};
use rsklearn_linear::Linear;
use rsklearn_zeror::ZeroRule;

pub struct DecisionTree<T> {
    metric: fn(Array<f64, Ix2>) -> f64,
    leaf: fn(Option<usize>, Option<f64>, Option<f64>) -> T,
    left: Option<T>,
    right: Option<T>,
    left_node: Option<Box<DecisionTree<T>>>,
    right_node: Option<Box<DecisionTree<T>>>,
    feat_index: usize,
    feat_val: f64,
    score: f64,
    epochs: Option<usize>,
    lr: Option<f64>,
    early_stop: Option<f64>,
    max_depth: usize,
    depth: usize,
}

impl<'a> Default for DecisionTree<ZeroRule> {
    fn default() -> Self {
        DecisionTree {
            metric: gini,
            leaf: zero_rule_leaf,
            left: None,
            right: None,
            left_node: None,
            right_node: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
            epochs: None,
            lr: None,
            early_stop: None,
            max_depth: 5,
            depth: 1,
        }
    }
}

impl<'a> DecisionTree<ZeroRule> {
    pub fn new(
        metric: fn(Array<f64, Ix2>) -> f64,
        max_depth: usize,
        depth: usize,
    ) -> DecisionTree<ZeroRule> {
        DecisionTree {
            metric: metric,
            leaf: zero_rule_leaf,
            left: None,
            right: None,
            left_node: None,
            right_node: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
            epochs: None,
            lr: None,
            early_stop: None,
            max_depth: max_depth,
            depth: depth,
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
                for (i, l_elm) in l.iter().enumerate() {
                    y_l.slice_mut(s![i, ..]).assign(&y.slice(s![*l_elm, ..]));
                }
                let mut y_r = Array::<f64, Ix2>::zeros((r.len(), y.shape()[1]));
                for (i, r_elm) in r.iter().enumerate() {
                    y_r.slice_mut(s![i, ..]).assign(&y.slice(s![*r_elm, ..]));
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

    pub fn get_node(&mut self, x: Array<f64, Ix2>, y: Array<f64, Ix2>) -> DecisionTree<ZeroRule> {
        let node = DecisionTree::<ZeroRule>::new(self.metric, self.max_depth, self.depth).fit(x, y);
        node
    }

    pub fn fit(mut self, x: Array<f64, Ix2>, y: Array<f64, Ix2>) -> Self {
        let leaf = self.leaf;
        let (left, right) = self.split_tree(x.clone(), y.clone());
        if self.depth < self.max_depth {
            if left.len() > 0 {
                let mut x_left = Array::<f64, Ix2>::zeros((left.len(), x.clone().shape()[1]));
                for (i, left_elm) in left.iter().enumerate() {
                    x_left
                        .slice_mut(s![i, ..])
                        .assign(&x.clone().slice(s![*left_elm, ..]));
                }
                let mut y_left = Array::<f64, Ix2>::zeros((left.len(), y.clone().shape()[1]));
                for (i, left_elm) in left.iter().enumerate() {
                    y_left
                        .slice_mut(s![i, ..])
                        .assign(&y.clone().slice(s![*left_elm, ..]));
                }
                let node = self.get_node(x_left, y_left);
                self.left_node = Some(Box::new(node));
            }
            if right.len() > 0 {
                let mut x_right = Array::<f64, Ix2>::zeros((right.len(), x.clone().shape()[1]));
                for (i, right_elm) in right.iter().enumerate() {
                    x_right
                        .slice_mut(s![i, ..])
                        .assign(&x.clone().slice(s![*right_elm, ..]));
                }
                let mut y_right = Array::<f64, Ix2>::zeros((right.len(), y.clone().shape()[1]));
                for (i, right_elm) in right.iter().enumerate() {
                    y_right
                        .slice_mut(s![i, ..])
                        .assign(&y.clone().slice(s![*right_elm, ..]));
                }
                let node = self.get_node(x_right, y_right);
                self.right_node = Some(Box::new(node));
            }
        } else {
            if left.len() > 0 {
                let mut x_left = Array::<f64, Ix2>::zeros((left.len(), x.clone().shape()[1]));
                for (i, left_elm) in left.iter().enumerate() {
                    x_left
                        .slice_mut(s![i, ..])
                        .assign(&x.clone().slice(s![*left_elm, ..]));
                }
                let mut y_left = Array::<f64, Ix2>::zeros((left.len(), y.clone().shape()[1]));
                for (i, left_elm) in left.iter().enumerate() {
                    y_left
                        .slice_mut(s![i, ..])
                        .assign(&y.clone().slice(s![*left_elm, ..]));
                }
                self.left = Some(leaf(None, None, None).fit(x_left, y_left));
            }
            if right.len() > 0 {
                let mut x_right = Array::<f64, Ix2>::zeros((right.len(), x.clone().shape()[1]));
                for (i, right_elm) in right.iter().enumerate() {
                    x_right
                        .slice_mut(s![i, ..])
                        .assign(&x.clone().slice(s![*right_elm, ..]));
                }
                let mut y_right = Array::<f64, Ix2>::zeros((right.len(), y.clone().shape()[1]));
                for (i, right_elm) in right.iter().enumerate() {
                    y_right
                        .slice_mut(s![i, ..])
                        .assign(&y.clone().slice(s![*right_elm, ..]));
                }
                self.right = Some(leaf(None, None, None).fit(x_right, y_right));
            }
        }
        self
    }

    pub fn predict(&self, x: Array<f64, Ix2>) -> Option<Array<f64, Ix2>> {
        let feat = x.slice(s![.., self.feat_index]);
        let val = self.feat_val;
        let (l, r) = self.make_split(feat.to_owned(), val.to_owned());
        if l.len() > 0 && r.len() > 0 {
            let mut x_l = Array::<f64, Ix2>::zeros((l.len(), x.shape()[1]));
            for (i, l_elm) in l.iter().enumerate() {
                x_l.slice_mut(s![i, ..]).assign(&x.slice(s![*l_elm, ..]))
            }
            if self.left.as_ref().is_none() && self.left.as_ref().is_none() {
                let left = self.left_node.as_ref().unwrap().predict(x_l).unwrap();
                let mut x_r = Array::<f64, Ix2>::zeros((r.len(), x.shape()[1]));
                for (i, r_elm) in r.iter().enumerate() {
                    x_r.slice_mut(s![i, ..]).assign(&x.slice(s![*r_elm, ..]))
                }
                let right = self.right_node.as_ref().unwrap().predict(x_r).unwrap();
                let mut z = Array::<f64, Ix2>::zeros((x.shape()[0], left.shape()[1]));
                for (i, l_elm) in l.iter().enumerate() {
                    z.slice_mut(s![*l_elm, ..]).assign(&left.slice(s![i, ..]));
                }
                for (i, r_elm) in r.iter().enumerate() {
                    z.slice_mut(s![*r_elm, ..]).assign(&right.slice(s![i, ..]));
                }
                Some(z)
            } else {
                let left = self.left.as_ref().unwrap().predict(x_l);
                let mut x_r = Array::<f64, Ix2>::zeros((r.len(), x.shape()[1]));
                for (i, r_elm) in r.iter().enumerate() {
                    x_r.slice_mut(s![i, ..]).assign(&x.slice(s![*r_elm, ..]))
                }
                let right = self.right.as_ref().unwrap().predict(x_r);
                let mut z = Array::<f64, Ix2>::zeros((x.shape()[0], left.shape()[1]));
                for (i, l_elm) in l.iter().enumerate() {
                    z.slice_mut(s![*l_elm, ..]).assign(&left.slice(s![i, ..]));
                }
                for (i, r_elm) in r.iter().enumerate() {
                    z.slice_mut(s![*r_elm, ..]).assign(&right.slice(s![i, ..]));
                }
                Some(z)
            }
        } else if l.len() > 0 {
            Some(self.left.as_ref().unwrap().predict(x))
        } else if r.len() > 0 {
            Some(self.right.as_ref().unwrap().predict(x))
        } else {
            None
        }
    }
}
