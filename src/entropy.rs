use std::collections::{HashSet, HashMap};

use ndarray::{Array, Ix1, Ix2, Axis};
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

pub fn deviation_org(y: Array<f64, Ix2>) -> f64 {
    let d = y.mapv(|v| v - y.mean().unwrap());
    let s = d.mapv(|v| v.powi(2));
    s.mean().unwrap().sqrt()
}

pub fn deviation(y: Array<f64, Ix2>) -> f64 {
    deviation_org(y)
}

pub fn gini_org(y: Array<f64, Ix2>) -> f64 {
    let mut i = Array::<usize, Ix1>::zeros(y.shape()[1]);
    for cl in 0..y.shape()[1] {
        i.slice_mut(s![cl]).fill(y.slice(s![.., cl]).argmax().unwrap());
    }
    let mut clz = HashSet::new();
    let mut c = HashMap::new();
    for cl in i.iter() {
        clz.insert(cl);
        if c.contains_key(cl) {
            c.insert(cl, 1.0 as f64);
        } else {
            c.insert(cl, *c.get(cl).unwrap() + 1.0 as f64);
        }
        
    }
    let size = y.shape()[0] as f64;
    let mut score = 0.0 as f64;
    for val in clz.iter() {
        score += (*c.get(val).unwrap() / size).powi(2);
    }
    1.0 - score
}

pub fn gini(y: Array<f64, Ix2>) -> f64 {
    let m = y.sum_axis(Axis(0));
    let size = y.shape()[0] as f64;
    let mut e = Array::<f64, Ix1>::zeros(m.shape()[0]);
    for i in 0..m.shape()[0] {
        e.slice_mut(s![i]).fill((m[i] / size).powi(2));
    }
    1.0 - e.sum()
}

pub fn infgain_org(y: Array<f64, Ix2>) -> f64 {
    let mut i = Array::<usize, Ix1>::zeros(y.shape()[1]);
    for cl in 0..y.shape()[1] {
        i.slice_mut(s![cl]).fill(y.slice(s![.., cl]).argmax().unwrap());
    }
    let mut clz = HashSet::new();
    let mut c = HashMap::new();
    for cl in i.iter() {
        clz.insert(cl);
        if c.contains_key(cl) {
            c.insert(cl, 1.0 as f64);
        } else {
            c.insert(cl, *c.get(cl).unwrap() + 1.0 as f64);
        } 
    }
    let size = y.shape()[0] as f64;
    let mut score = 0.0 as f64;
    for val in clz.iter() {
        let p = *c.get(val).unwrap() / size;
        if p != 0.0 as f64 {
            score += p * p.log2();
        }
    }
    -score
}

pub fn infgain(y: Array<f64, Ix2>) -> f64 {
    let m = y.sum_axis(Axis(0));
    let size = y.shape()[0] as f64;
    let mut e = Array::<f64, Ix1>::zeros(m.shape()[0]);
    for i in 0..m.shape()[0] {
        if m[i] != 0.0 as f64 {
            e.slice_mut(s![i]).fill(m[i] * (m[i] / size).log2());
        }  
    }
    -e.sum() 
}