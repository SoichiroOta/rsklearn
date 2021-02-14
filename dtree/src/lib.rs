extern crate ndarray;
use ndarray::prelude::*;
use ndarray::{Array, Ix1, Ix2, Ix3};
use ndarray_stats::QuantileExt;

use rsklearn::entropy::gini;
use rsklearn_dstump::{linear_dstump, zero_rule_dstump, DecisionStump};
use rsklearn_linear::Linear;
use rsklearn_numrs::partial_argsort;
use rsklearn_zeror::ZeroRule;

pub struct DecisionTree<T> {
    metric: fn(Array<f64, Ix2>) -> f64,
    leaf:
        fn(fn(Array<f64, Ix2>) -> f64, Option<usize>, Option<f64>, Option<f64>) -> DecisionStump<T>,
    node: fn(
        fn(Array<f64, Ix2>) -> f64,
        Option<usize>,
        Option<f64>,
        Option<f64>,
        usize,
        usize,
    ) -> Box<DecisionTree<T>>,
    left: Option<DecisionStump<T>>,
    right: Option<DecisionStump<T>>,
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

pub fn zero_rule_dstump_leaf(
    metric: fn(Array<f64, Ix2>) -> f64,
    _epochs: Option<usize>,
    _lr: Option<f64>,
    _early_stop: Option<f64>,
) -> DecisionStump<ZeroRule> {
    zero_rule_dstump(metric)
}

pub fn zero_rule_dtree(
    metric: fn(Array<f64, Ix2>) -> f64,
    max_depth: usize,
    depth: usize,
) -> DecisionTree<ZeroRule> {
    DecisionTree::<ZeroRule>::new(metric, max_depth, depth)
}

pub fn zero_rule_dtree_node(
    metric: fn(Array<f64, Ix2>) -> f64,
    _epochs: Option<usize>,
    _lr: Option<f64>,
    _early_stop: Option<f64>,
    max_depth: usize,
    depth: usize,
) -> Box<DecisionTree<ZeroRule>> {
    Box::new(zero_rule_dtree(metric, max_depth, depth))
}

impl<'a> Default for DecisionTree<ZeroRule> {
    fn default() -> Self {
        DecisionTree {
            metric: gini,
            leaf: zero_rule_dstump_leaf,
            node: zero_rule_dtree_node,
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
            leaf: zero_rule_dstump_leaf,
            node: zero_rule_dtree_node,
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

    pub fn split_tree_slow(
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

    pub fn split_tree_fast(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix2>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.feat_index = 0;
        self.feat_val = f64::INFINITY;
        let mut score = f64::INFINITY;
        let mut xindex = Array::<usize, Ix2>::zeros((x.shape()[0], x.shape()[1]));
        for i in 0..x.shape()[1] {
            let mut x_col = Array::<f64, Ix1>::zeros(x.shape()[0]);
            x_col.assign(&x.slice(s![.., i]));
            let x_col_index = partial_argsort(&x_col.clone(), false);
            xindex.slice_mut(s![.., i]).assign(&x_col_index);
        }
        let mut ysot = Array::<f64, Ix3>::zeros((y.shape()[0], y.shape()[1], xindex.shape()[1]));
        for j in 0..xindex.shape()[1] {
            for i in 0..xindex.shape()[0] {
                ysot.slice_mut(s![i, .., j])
                    .assign(&y.slice(s![xindex[[i, j]], ..]));
            }
        }
        for f in 1..x.shape()[0] {
            let ly = ysot.slice(s![..f, .., ..]);
            let ry = ysot.slice(s![f.., .., ..]);
            let mut loss = Array::<f64, Ix1>::zeros(x.shape()[1]);
            for yp in 0..x.shape()[1] {
                let loss_elm = if x[[xindex[[f - 1, yp]], yp]] != x[[xindex[[f, yp]], yp]] {
                    self.make_loss(
                        ly.slice(s![.., .., yp]).to_owned(),
                        ry.slice(s![.., .., yp]).to_owned(),
                    )
                } else {
                    f64::INFINITY
                };
                loss.slice_mut(s![yp]).fill(loss_elm);
            }
            let i = loss.argmin().unwrap();
            if score > loss[[i]] {
                score = loss[[i]];
                self.feat_index = i;
                self.feat_val = x[[xindex[[f, i]], i]];
            }
        }
        let filter = x.slice(s![.., self.feat_index]).mapv(|a| a < self.feat_val);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, f) in filter.iter().enumerate() {
            if *f == true {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        self.score = score;
        (left, right)
    }

    pub fn split_tree(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix2>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.split_tree_fast(x, y)
    }

    pub fn get_node(&mut self, x: Array<f64, Ix2>, y: Array<f64, Ix2>) -> DecisionTree<ZeroRule> {
        let node = self.node;
        node(
            self.metric,
            None,
            None,
            None,
            self.max_depth,
            self.depth + 1,
        )
        .fit(x, y)
    }

    pub fn fit(mut self, x: Array<f64, Ix2>, y: Array<f64, Ix2>) -> Self {
        let leaf = self.leaf;
        let (left, right) = self.split_tree(x.clone(), y.clone());
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
        if self.depth < self.max_depth && left.len() > 0 {
            let node = self.get_node(x_left, y_left);
            self.left_node = Some(Box::new(node));
        } else {
            self.left = Some(leaf(self.metric, None, None, None).fit(x_left, y_left));
        }
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
        if self.depth < self.max_depth && right.len() > 0 {
            let node = self.get_node(x_right, y_right);
            self.right_node = Some(Box::new(node));
        } else {
            self.right = Some(leaf(self.metric, None, None, None).fit(x_right, y_right));
        }
        self
    }

    pub fn predict_with_left(&self, x_l: Array<f64, Ix2>) -> Array<f64, Ix2> {
        if self.left.as_ref().is_none() {
            self.left_node.as_ref().unwrap().predict(x_l).unwrap()
        } else {
            self.left.as_ref().unwrap().predict(x_l).unwrap()
        }
    }

    pub fn predict_with_right(&self, x_r: Array<f64, Ix2>) -> Array<f64, Ix2> {
        if self.right.as_ref().is_none() {
            self.right_node.as_ref().unwrap().predict(x_r).unwrap()
        } else {
            self.right.as_ref().unwrap().predict(x_r).unwrap()
        }
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
            let left = self.predict_with_left(x_l);
            let mut x_r = Array::<f64, Ix2>::zeros((r.len(), x.shape()[1]));
            for (i, r_elm) in r.iter().enumerate() {
                x_r.slice_mut(s![i, ..]).assign(&x.slice(s![*r_elm, ..]))
            }
            let right = self.predict_with_right(x_r);
            let mut z = Array::<f64, Ix2>::zeros((x.shape()[0], left.shape()[1]));
            for (i, l_elm) in l.iter().enumerate() {
                z.slice_mut(s![*l_elm, ..]).assign(&left.slice(s![i, ..]));
            }
            for (i, r_elm) in r.iter().enumerate() {
                z.slice_mut(s![*r_elm, ..]).assign(&right.slice(s![i, ..]));
            }
            Some(z)
        } else if l.len() > 0 {
            Some(self.predict_with_left(x))
        } else if r.len() > 0 {
            Some(self.predict_with_right(x))
        } else {
            None
        }
    }

    pub fn print_leaf(
        &self,
        tree: Option<&Box<DecisionTree<ZeroRule>>>,
        stump: Option<&DecisionStump<ZeroRule>>,
        d: Option<usize>,
    ) -> String {
        let d_ = if d.is_none() { 0 } else { d.unwrap() };
        if stump.is_none() {
            let mut s = Vec::new();
            let mut pluses = String::from("");
            for _ in 0..d_ {
                pluses = pluses + &String::from("+");
            }
            s.push(format!(
                "{}if feat[ {} ] <= {} then:\n",
                pluses,
                self.feat_index.to_string(),
                self.feat_val.to_string()
            ));
            if tree.is_none() {
                if self.left.is_none() {
                    let left = self.left_node.as_ref();
                    s.push(self.print_leaf(Some(left.unwrap()), None, Some(d_ + 1)));
                } else {
                    s.push(self.print_leaf(None, Some(self.left.as_ref().unwrap()), Some(d_ + 1)));
                }
                let mut bars = String::from("");
                for _ in 0..d_ {
                    bars = bars + &String::from("|");
                }
                s.push(format!("{}else\n", bars));
                if self.right.is_none() {
                    let right = self.right_node.as_ref();
                    s.push(self.print_leaf(Some(right.unwrap()), None, Some(d_ + 1)));
                } else {
                    s.push(self.print_leaf(None, Some(self.right.as_ref().unwrap()), Some(d_ + 1)));
                }
            } else {
                let tree_ = tree.unwrap();
                if tree_.left.is_none() {
                    let left = tree_.left_node.as_ref();
                    s.push(self.print_leaf(Some(&left.unwrap()), None, Some(d_ + 1)));
                } else {
                    let left = tree_.left.as_ref();
                    s.push(self.print_leaf(None, Some(&left.unwrap()), Some(d_ + 1)));
                }
                let mut bars = String::from("");
                for _ in 0..d_ {
                    bars = bars + &String::from("|");
                }
                s.push(format!("{}else\n", bars));
                if tree_.right.is_none() {
                    let right = tree_.right_node.as_ref();
                    s.push(self.print_leaf(Some(&right.unwrap()), None, Some(d_ + 1)));
                } else {
                    let right = tree_.right.as_ref();
                    s.push(self.print_leaf(None, Some(&right.unwrap()), Some(d_ + 1)));
                }
            };
            String::from(s.iter().cloned().collect::<String>())
        } else {
            let mut bars = String::from("");
            for _ in 0..d_ - 1 {
                bars = bars + &String::from("|");
            }
            format!("{} {}", bars, stump.unwrap().str())
        }
    }

    pub fn str(self) -> String {
        self.print_leaf(None, None, None)
    }
}

pub fn linear_dstump_leaf(
    metric: fn(Array<f64, Ix2>) -> f64,
    epochs: Option<usize>,
    lr: Option<f64>,
    early_stop: Option<f64>,
) -> DecisionStump<Linear> {
    linear_dstump(metric, epochs, lr, early_stop)
}

pub fn linear_dtree(
    metric: fn(Array<f64, Ix2>) -> f64,
    epochs: Option<usize>,
    lr: Option<f64>,
    early_stop: Option<f64>,
    max_depth: usize,
    depth: usize,
) -> DecisionTree<Linear> {
    DecisionTree::<Linear>::new(metric, epochs, lr, early_stop, max_depth, depth)
}

pub fn linear_dtree_node(
    metric: fn(Array<f64, Ix2>) -> f64,
    epochs: Option<usize>,
    lr: Option<f64>,
    early_stop: Option<f64>,
    max_depth: usize,
    depth: usize,
) -> Box<DecisionTree<Linear>> {
    Box::new(linear_dtree(
        metric, epochs, lr, early_stop, max_depth, depth,
    ))
}

impl<'a> Default for DecisionTree<Linear> {
    fn default() -> Self {
        DecisionTree {
            metric: gini,
            leaf: linear_dstump_leaf,
            node: linear_dtree_node,
            left: None,
            right: None,
            left_node: None,
            right_node: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
            epochs: Some(20),
            lr: Some(0.01),
            early_stop: None,
            max_depth: 5,
            depth: 1,
        }
    }
}

impl<'a> DecisionTree<Linear> {
    pub fn new(
        metric: fn(Array<f64, Ix2>) -> f64,
        epochs: Option<usize>,
        lr: Option<f64>,
        early_stop: Option<f64>,
        max_depth: usize,
        depth: usize,
    ) -> DecisionTree<Linear> {
        DecisionTree {
            metric: metric,
            leaf: linear_dstump_leaf,
            node: linear_dtree_node,
            left: None,
            right: None,
            left_node: None,
            right_node: None,
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
            epochs: epochs,
            lr: lr,
            early_stop: early_stop,
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

    pub fn make_loss(&self, y1: Array<f64, Ix1>, y2: Array<f64, Ix1>) -> f64 {
        if y1.shape()[0] == 0 || y2.shape()[0] == 0 {
            return f64::INFINITY;
        }
        let total = y1.shape()[0] as f64 + y2.shape()[0] as f64;
        let metric = self.metric;
        let mut y1_ix2 = Array::<f64, Ix2>::zeros((y1.shape()[0], 1));
        y1_ix2.slice_mut(s![.., 0]).assign(&y1);
        let m1 = metric(y1_ix2.clone()) * (y1.shape()[0] as f64 / total);
        let mut y2_ix2 = Array::<f64, Ix2>::zeros((y2.shape()[0], 1));
        y2_ix2.slice_mut(s![.., 0]).assign(&y2);
        let m2 = metric(y2_ix2.clone()) * (y2.shape()[0] as f64 / total);
        m1 + m2
    }

    pub fn split_tree_slow(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix1>,
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
                let mut y_l = Array::<f64, Ix1>::zeros(l.len());
                for (i, l_elm) in l.iter().enumerate() {
                    y_l.slice_mut(s![i]).assign(&y.slice(s![*l_elm]));
                }
                let mut y_r = Array::<f64, Ix1>::zeros(r.len());
                for (i, r_elm) in r.iter().enumerate() {
                    y_r.slice_mut(s![i]).assign(&y.slice(s![*r_elm]));
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

    pub fn split_tree_fast(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix1>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.feat_index = 0;
        self.feat_val = f64::INFINITY;
        let mut score = f64::INFINITY;
        let mut xindex = Array::<usize, Ix2>::zeros((x.shape()[0], x.shape()[1]));
        for i in 0..x.shape()[1] {
            let mut x_col = Array::<f64, Ix1>::zeros(x.shape()[0]);
            x_col.assign(&x.slice(s![.., i]));
            let x_col_index = partial_argsort(&x_col.clone(), false);
            xindex.slice_mut(s![.., i]).assign(&x_col_index);
        }
        let mut ysot = Array::<f64, Ix2>::zeros((y.shape()[0], xindex.shape()[1]));
        for j in 0..xindex.shape()[1] {
            for i in 0..xindex.shape()[0] {
                ysot.slice_mut(s![i, j])
                    .assign(&y.slice(s![xindex[[i, j]]]));
            }
        }
        for f in 1..x.shape()[0] {
            let ly = ysot.slice(s![..f, ..]);
            let ry = ysot.slice(s![f.., ..]);
            let mut loss = Array::<f64, Ix1>::zeros(x.shape()[1]);
            for yp in 0..x.shape()[1] {
                let loss_elm = if x[[xindex[[f - 1, yp]], yp]] != x[[xindex[[f, yp]], yp]] {
                    self.make_loss(
                        ly.slice(s![.., yp]).to_owned(),
                        ry.slice(s![.., yp]).to_owned(),
                    )
                } else {
                    f64::INFINITY
                };
                loss.slice_mut(s![yp]).fill(loss_elm);
            }
            let i = loss.argmin().unwrap();
            if score > loss[[i]] {
                score = loss[[i]];
                self.feat_index = i;
                self.feat_val = x[[xindex[[f, i]], i]];
            }
        }
        let filter = x.slice(s![.., self.feat_index]).mapv(|a| a < self.feat_val);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, f) in filter.iter().enumerate() {
            if *f == true {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        self.score = score;
        (left, right)
    }

    pub fn split_tree(
        &mut self,
        x: Array<f64, Ix2>,
        y: Array<f64, Ix1>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.split_tree_fast(x, y)
    }

    pub fn get_node(&mut self, x: Array<f64, Ix2>, y: Array<f64, Ix1>) -> DecisionTree<Linear> {
        let node = self.node;
        node(
            self.metric,
            self.epochs,
            self.lr,
            self.early_stop,
            self.max_depth,
            self.depth + 1,
        )
        .fit(x, y)
    }

    pub fn fit(mut self, x: Array<f64, Ix2>, y: Array<f64, Ix1>) -> Self {
        let leaf = self.leaf;
        let (left, right) = self.split_tree(x.clone(), y.clone());
        let mut x_left = Array::<f64, Ix2>::zeros((left.len(), x.clone().shape()[1]));
        for (i, left_elm) in left.iter().enumerate() {
            x_left
                .slice_mut(s![i, ..])
                .assign(&x.clone().slice(s![*left_elm, ..]));
        }
        let mut y_left = Array::<f64, Ix1>::zeros(left.len());
        for (i, left_elm) in left.iter().enumerate() {
            y_left
                .slice_mut(s![i])
                .assign(&y.clone().slice(s![*left_elm]));
        }
        if self.depth < self.max_depth && left.len() > 0 {
            let node = self.get_node(x_left, y_left);
            self.left_node = Some(Box::new(node));
        } else {
            self.left =
                Some(leaf(self.metric, self.epochs, self.lr, self.early_stop).fit(x_left, y_left));
        }
        let mut x_right = Array::<f64, Ix2>::zeros((right.len(), x.clone().shape()[1]));
        for (i, right_elm) in right.iter().enumerate() {
            x_right
                .slice_mut(s![i, ..])
                .assign(&x.clone().slice(s![*right_elm, ..]));
        }
        let mut y_right = Array::<f64, Ix1>::zeros(right.len());
        for (i, right_elm) in right.iter().enumerate() {
            y_right
                .slice_mut(s![i])
                .assign(&y.clone().slice(s![*right_elm]));
        }
        if self.depth < self.max_depth && right.len() > 0 {
            let node = self.get_node(x_right, y_right);
            self.right_node = Some(Box::new(node));
        } else {
            self.right = Some(
                leaf(self.metric, self.epochs, self.lr, self.early_stop).fit(x_right, y_right),
            );
        }
        self
    }

    pub fn predict_with_left(&self, x_l: Array<f64, Ix2>) -> Array<f64, Ix1> {
        if self.left.as_ref().is_none() {
            self.left_node.as_ref().unwrap().predict(x_l).unwrap()
        } else {
            self.left.as_ref().unwrap().predict(x_l).unwrap()
        }
    }

    pub fn predict_with_right(&self, x_r: Array<f64, Ix2>) -> Array<f64, Ix1> {
        if self.right.as_ref().is_none() {
            self.right_node.as_ref().unwrap().predict(x_r).unwrap()
        } else {
            self.right.as_ref().unwrap().predict(x_r).unwrap()
        }
    }

    pub fn predict(&self, x: Array<f64, Ix2>) -> Option<Array<f64, Ix1>> {
        let feat = x.slice(s![.., self.feat_index]);
        let val = self.feat_val;
        let (l, r) = self.make_split(feat.to_owned(), val.to_owned());
        if l.len() > 0 && r.len() > 0 {
            let mut x_l = Array::<f64, Ix2>::zeros((l.len(), x.shape()[1]));
            for (i, l_elm) in l.iter().enumerate() {
                x_l.slice_mut(s![i, ..]).assign(&x.slice(s![*l_elm, ..]))
            }
            let left = self.predict_with_left(x_l);
            let mut x_r = Array::<f64, Ix2>::zeros((r.len(), x.shape()[1]));
            for (i, r_elm) in r.iter().enumerate() {
                x_r.slice_mut(s![i, ..]).assign(&x.slice(s![*r_elm, ..]))
            }
            let right = self.predict_with_right(x_r);
            let mut z = Array::<f64, Ix1>::zeros(x.shape()[0]);
            for (i, l_elm) in l.iter().enumerate() {
                z.slice_mut(s![*l_elm]).assign(&left.slice(s![i]));
            }
            for (i, r_elm) in r.iter().enumerate() {
                z.slice_mut(s![*r_elm]).assign(&right.slice(s![i]));
            }
            Some(z)
        } else if l.len() > 0 {
            Some(self.predict_with_left(x))
        } else if r.len() > 0 {
            Some(self.predict_with_right(x))
        } else {
            None
        }
    }

    pub fn print_leaf(
        &self,
        tree: Option<&Box<DecisionTree<Linear>>>,
        stump: Option<&DecisionStump<Linear>>,
        d: Option<usize>,
    ) -> String {
        let d_ = if d.is_none() { 0 } else { d.unwrap() };
        if stump.is_none() {
            let mut s = Vec::new();
            let mut pluses = String::from("");
            for _ in 0..d_ {
                pluses = pluses + &String::from("+");
            }
            s.push(format!(
                "{}if feat[ {} ] <= {} then:\n",
                pluses,
                self.feat_index.to_string(),
                self.feat_val.to_string()
            ));
            if tree.is_none() {
                if self.left.is_none() {
                    let left = self.left_node.as_ref();
                    s.push(self.print_leaf(Some(left.unwrap()), None, Some(d_ + 1)));
                } else {
                    s.push(self.print_leaf(None, Some(self.left.as_ref().unwrap()), Some(d_ + 1)));
                }
                let mut bars = String::from("");
                for _ in 0..d_ {
                    bars = bars + &String::from("|");
                }
                s.push(format!("{}else\n", bars));
                if self.right.is_none() {
                    let right = self.right_node.as_ref();
                    s.push(self.print_leaf(Some(right.unwrap()), None, Some(d_ + 1)));
                } else {
                    s.push(self.print_leaf(None, Some(self.right.as_ref().unwrap()), Some(d_ + 1)));
                }
            } else {
                let tree_ = tree.unwrap();
                if tree_.left.is_none() {
                    let left = tree_.left_node.as_ref();
                    s.push(self.print_leaf(Some(&left.unwrap()), None, Some(d_ + 1)));
                } else {
                    let left = tree_.left.as_ref();
                    s.push(self.print_leaf(None, Some(&left.unwrap()), Some(d_ + 1)));
                }
                let mut bars = String::from("");
                for _ in 0..d_ {
                    bars = bars + &String::from("|");
                }
                s.push(format!("{}else\n", bars));
                if tree_.right.is_none() {
                    let right = tree_.right_node.as_ref();
                    s.push(self.print_leaf(Some(&right.unwrap()), None, Some(d_ + 1)));
                } else {
                    let right = tree_.right.as_ref();
                    s.push(self.print_leaf(None, Some(&right.unwrap()), Some(d_ + 1)));
                }
            };
            String::from(s.iter().cloned().collect::<String>())
        } else {
            let mut bars = String::from("");
            for _ in 0..d_ - 1 {
                bars = bars + &String::from("|");
            }
            format!("{} {}", bars, stump.unwrap().str())
        }
    }

    pub fn str(self) -> String {
        self.print_leaf(None, None, None)
    }
}
