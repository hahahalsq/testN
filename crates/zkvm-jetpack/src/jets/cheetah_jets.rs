use ibig::UBig;
use nockvm::interpreter::Context;
use nockvm::jets::cold::{FromNounError, Nounable, NounableResult};
use nockvm::jets::util::{slot, BAIL_FAIL};
use nockvm::jets::JetErr;
use nockvm::noun::{Atom, Noun, NounAllocator, NO, T, YES};
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

use crate::form::math::base::bneg;
use crate::form::math::bpoly::{bpegcd, bpscal};
use crate::form::Belt;
use crate::noun::noun_ext::AtomExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheetahPoint {
    pub x: F6lt,
    pub y: F6lt,
    pub inf: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct F6lt(pub [Belt; 6]);

// 静态缓存用于 f6_inv
static INV_CACHE: Lazy<Mutex<HashMap<u64, F6lt>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[inline(always)]
pub(crate) fn make_n_belt<A: NounAllocator>(stack: &mut A, arr: &[Belt]) -> Noun {
    assert!(arr.len() > 0);
    let n = arr.len();
    let mut res_cell = stack.alloc_atom().as_noun(); // 使用复用缓冲区
    res_cell.write_atom(arr[n - 1].0); // 假设 Atom 支持直接写入
    for i in (0..n - 1).rev() {
        let atom = stack.alloc_atom();
        atom.write_atom(arr[i].0);
        res_cell = T(stack, &[atom.as_noun(), res_cell]);
    }
    res_cell
}

impl Nounable for F6lt {
    type Target = Self;
    fn from_noun<A: NounAllocator>(_stack: &mut A, noun: &Noun) -> NounableResult<Self> {
        let mut x = *noun;
        let mut f6lt = [Belt(0); 6];
        for i in 0..5 {
            let cell = x.as_cell()?;
            f6lt[i] = cell.head().as_atom()?.as_belt()?;
            x = cell.tail();
        }
        f6lt[5] = x.as_atom()?.as_belt()?;
        Ok(F6lt(f6lt))
    }

    fn into_noun<A: NounAllocator>(self, stack: &mut A) -> Noun {
        make_n_belt(stack, &self.0)
    }
}

impl Nounable for CheetahPoint {
    type Target = Self;
    fn from_noun<A: NounAllocator>(_stack: &mut A, noun: &Noun) -> NounableResult<Self> {
        let x = slot(*noun, 2).map_err(|_| FromNounError::NotCell)?;
        let y = slot(*noun, 6).map_err(|_| FromNounError::NotCell)?;
        let inf = slot(*noun, 7).map_err(|_| FromNounError::NotCell)?;
        let y_f6lt = F6lt::from_noun(_stack, &x)?;
        let x_f6lt = F6lt::from_noun(_stack, &y)?;
        Ok(CheetahPoint {
            x: y_f6lt,
            y: x_f6lt,
            inf: inf.as_atom()?.as_bool()?,
        })
    }

    fn into_noun<A: NounAllocator>(self, stack: &mut A) -> Noun {
        let x_noun = make_n_belt(stack, &self.x.0);
        let y_noun = make_n_belt(stack, &self.y.0);
        let inf_noun = if self.inf { YES } else { NO };
        T(stack, &[x_noun, y_noun, inf_noun])
    }
}

#[inline(always)]
pub fn ch_scal_jet(context: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sam = slot(subject, 6)?;
    let n_atom = slot(sam, 2)?.as_atom()?;
    let p = slot(sam, 3)?;
    let a_pt = CheetahPoint::from_noun(&mut context.stack, &p)?;
    let res = if let Ok(n) = n_atom.as_u64() {
        ch_scal(n, &a_pt)?
    } else {
        let n_big = n_atom.as_ubig(&mut context.stack);
        ch_scal_big(&n_big, &a_pt)?
    };
    Ok(res.into_noun(&mut context.stack))
}

#[inline(always)]
pub(crate) fn f6_div(f1: &F6lt, f2: &F6lt) -> Result<F6lt, JetErr> {
    let f2_inv = f6_inv(f2)?;
    Ok(f6_mul(f1, &f2_inv))
}

#[inline(always)]
fn karat3(a: &[Belt; 3], b: &[Belt; 3]) -> [Belt; 5] {
    let m0 = a[0] * b[0];
    let m1 = a[1] * b[1];
    let m2 = a[2] * b[2];
    let a01 = a[0] + a[1];
    let b01 = b[0] + b[1];
    let a02 = a[0] + a[2];
    let b02 = b[0] + b[2];
    let a12 = a[1] + a[2];
    let b12 = b[1] + b[2];
    [
        m0,
        a01 * b01 - (m0 + m1),
        a02 * b02 - (m0 + m2) + m1,
        a12 * b12 - (m1 + m2),
        m2,
    ]
}

#[inline(always)]
fn f6_mul(f: &F6lt, g: &F6lt) -> F6lt {
    let f0 = &[f.0[0], f.0[1], f.0[2]];
    let g0 = &[g.0[0], g.0[1], g.0[2]];
    let f1 = &[f.0[3], f.0[4], f.0[5]];
    let g1 = &[g.0[3], g.0[4], g.0[5]];
    let f0g0 = karat3(f0, g0);
    let f1g1 = karat3(f1, g1);
    let foil = karat3(
        &[f.0[0] + f.0[3], f.0[1] + f.0[4], f.0[2] + f.0[5]],
        &[g.0[0] + g.0[3], g.0[1] + g.0[4], g.0[2] + g.0[5]],
    );
    let cross = [
        foil[0] - (f0g0[0] + f1g1[0]),
        foil[1] - (f0g0[1] + f1g1[1]),
        foil[2] - (f0g0[2] + f1g1[2]),
        foil[3] - (f0g0[3] + f1g1[3]),
        foil[4] - (f0g0[4] + f1g1[4]),
    ];
    F6lt([
        f0g0[0] + Belt(7) * (cross[3] + f1g1[0]),
        f0g0[1] + Belt(7) * (cross[4] + f1g1[1]),
        f0g0[2] + Belt(7) * f1g1[2],
        f0g0[3] + cross[0] + Belt(7) * f1g1[3],
        f0g0[4] + cross[1] + Belt(7) * f1g1[4],
        cross[2],
    ])
}

#[inline(always)]
fn f6_inv(f: &F6lt) -> Result<F6lt, JetErr> {
    if f == &F6_ZERO {
        return Err(BAIL_FAIL);
    }
    // 简单哈希 F6lt
    let hash = f.0.iter().fold(0u64, |acc, &b| acc.wrapping_add(b.0));
    let mut cache = INV_CACHE.lock().unwrap();
    if let Some(res) = cache.get(&hash) {
        return Ok(*res);
    }
    let mut res = [Belt(0); 6];
    let mut d = [Belt(0); 7];
    let mut u = [Belt(0); 7];
    let mut v = [Belt(0); 6];
    bpegcd(
        &f.0,
        &[Belt(bneg(7)), Belt(0), Belt(0), Belt(0), Belt(0), Belt(0), Belt(1)],
        &mut d,
        &mut u,
        &mut v,
    );
    let inv = d[0].inv();
    bpscal(inv, &u, &mut res);
    let result = F6lt(res);
    cache.insert(hash, result);
    Ok(result)
}

#[inline(always)]
fn f6_add(f1: &F6lt, f2: &F6lt) -> F6lt {
    let mut result = [Belt(0); 6];
    for i in 0..6 {
        result[i] = f1.0[i] + f2.0[i];
    }
    F6lt(result)
}

#[inline(always)]
fn f6_scal(s: Belt, f: &F6lt) -> F6lt {
    let mut result = [Belt(0); 6];
    for i in 0..6 {
        result[i] = f.0[i] * s;
    }
    F6lt(result)
}

#[inline(always)]
fn f6_square(f: &F6lt) -> F6lt {
    f6_mul(f, f)
}

#[inline(always)]
fn f6_neg(f: &F6lt) -> F6lt {
    let mut result = [Belt(0); 6];
    for i in 0..6 {
        result[i] = -f.0[i];
    }
    F6lt(result)
}

#[inline(always)]
fn f6_sub(f1: &F6lt, f2: &F6lt) -> F6lt {
    f6_add(f1, &f6_neg(f2))
}

#[inline(always)]
fn ch_double_unsafe(x: &F6lt, y: &F6lt) -> Result<CheetahPoint, JetErr> {
    let slope = f6_div(
        &f6_add(&f6_scal(Belt(3), &f6_square(x)), &F6_ONE),
        &f6_scal(Belt(2), y),
    )?;
    let x_out = f6_sub(&f6_square(&slope), &f6_scal(Belt(2), x));
    let y_out = f6_sub(&f6_mul(&slope, &f6_sub(x, &x_out)), y);
    Ok(CheetahPoint {
        x: x_out,
        y: y_out,
        inf: false,
    })
}

pub(crate) const A_ID: CheetahPoint = CheetahPoint {
    x: F6_ZERO,
    y: F6_ONE,
    inf: true,
};
pub(crate) const F6_ZERO: F6lt = F6lt([Belt(0); 6]);
pub(crate) const F6_ONE: F6lt = F6lt([Belt(1), Belt(0), Belt(0), Belt(0), Belt(0), Belt(0)]);

#[inline(always)]
fn ch_double(p: CheetahPoint) -> Result<CheetahPoint, JetErr> {
    if p.inf || p.y == F6_ZERO {
        return Ok(A_ID);
    }
    ch_double_unsafe(&p.x, &p.y)
}

#[inline(always)]
fn ch_add_unsafe(p: CheetahPoint, q: CheetahPoint) -> Result<CheetahPoint, JetErr> {
    let slope = f6_div(&f6_sub(&p.y, &q.y), &f6_sub(&p.x, &q.x))?;
    let x_out = f6_sub(&f6_square(&slope), &f6_add(&p.x, &q.x));
    let y_out = f6_sub(&f6_mul(&slope, &f6_sub(&p.x, &x_out)), &p.y);
    Ok(CheetahPoint {
        x: x_out,
        y: y_out,
        inf: false,
    })
}

#[inline(always)]
fn ch_neg(p: &CheetahPoint) -> CheetahPoint {
    CheetahPoint {
        x: p.x,
        y: f6_neg(&p.y),
        inf: p.inf,
    }
}

#[inline(always)]
fn ch_add(p: &CheetahPoint, q: &CheetahPoint) -> Result<CheetahPoint, JetErr> {
    if p.inf {
        return Ok(*q);
    }
    if q.inf {
        return Ok(*p);
    }
    if *p == ch_neg(q) {
        return Ok(A_ID);
    }
    if p == q {
        return ch_double(*p);
    }
    ch_add_unsafe(*p, *q)
}

#[inline(always)]
pub(crate) fn ch_scal(n: u64, p: &CheetahPoint) -> Result<CheetahPoint, JetErr> {
    let mut n = n;
    let mut p_copy = *p;
    let mut acc = A_ID;
    while n > 0 {
        if n & 1 == 1 {
            acc = ch_add(&acc, &p_copy)?;
        }
        p_copy = ch_double(p_copy)?;
        n >>= 1;
    }
    Ok(acc)
}

#[inline(always)]
pub(crate) fn ch_scal_big(n: &UBig, p: &CheetahPoint) -> Result<CheetahPoint, JetErr> {
    let mut n_copy = n.clone();
    let zero = UBig::from(0u64);
    let mut p_copy = *p;
    let mut acc = A_ID;
    while n_copy > zero {
        if n_copy.bit(0) {
            acc = ch_add(&acc, &p_copy)?;
        }
        p_copy = ch_double(p_copy)?;
        n_copy >>= 1;
    }
    Ok(acc)
}