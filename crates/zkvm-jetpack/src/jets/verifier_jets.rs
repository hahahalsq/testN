use nockvm::interpreter::Context;
use nockvm::jets::util::slot;
use nockvm::jets::JetErr;
use nockvm::noun::{Cell, IndirectAtom, Noun};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::sync::Mutex;
use std::thread;
use once_cell::sync::Lazy;

use crate::form::math::fext::*;
use crate::form::poly::Poly;
use crate::form::{Belt, FPolySlice, Felt};
use crate::hand::handle::new_handle_mut_felt;
use crate::hand::structs::HoonList;
use crate::jets::utils::jet_err;
use crate::noun::noun_ext::NounExt;

// Static cache for ordered_root results
static ROOT_CACHE: Lazy<Mutex<HashMap<u64, Belt>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub fn evaluate_deep_jet(context: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sam = slot(subject, 6)?;
    let mut sam_cur: Cell = sam.as_cell()?;

    // Extract parameters with minimal allocations
    let trace_evaluations = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let comp_evaluations = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let trace_elems_noun = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let comp_elems_noun = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let num_comp_pieces = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let weights = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let heights_noun = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let full_widths_noun = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let omega = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let index = sam_cur.head();
    sam_cur = sam_cur.tail().as_cell()?;
    let deep_challenge = sam_cur.head();
    let new_comp_eval = sam_cur.tail();

    // Convert nouns to appropriate types with optimized allocation
    let trace_evaluations = FPolySlice::try_from(trace_evaluations)?;
    let comp_evaluations = FPolySlice::try_from(comp_evaluations)?;
    let num_comp_pieces = num_comp_pieces.as_atom()?.as_u64()?;
    let weights = FPolySlice::try_from(weights)?;
    let omega = omega.as_felt()?;
    let index = index.as_atom()?.as_u64()?;
    let deep_challenge = deep_challenge.as_felt()?;
    let new_comp_eval = new_comp_eval.as_felt()?;

    // Optimized HoonList to Vec<Belt> conversion
    let trace_elems = noun_to_belt_vec(context, trace_elems_noun)?;
    let comp_elems = noun_to_belt_vec(context, comp_elems_noun)?;
    let heights = noun_to_u64_vec(context, heights_noun)?;
    let full_widths = noun_to_u64_vec(context, full_widths_noun)?;

    // Precompute omicron values and omega_pow
    let g = Felt::lift(Belt(7));
    let omega_pow = fmul_(&fpow_(&omega, index as u64), &g);
    let omicrons: Vec<Felt> = heights
        .iter()
        .map(|&h| {
            let mut cache = ROOT_CACHE.lock().unwrap();
            let belt = cache
                .entry(h)
                .or_insert_with(|| Belt(h).ordered_root().unwrap());
            Ok(Felt::lift(*belt))
        })
        .collect::<Result<Vec<Felt>, JetErr>>()?;

    // Parallelize heights loop
    let (tx, rx): (Sender<(Felt, usize)>, Receiver<(Felt, usize)>) = channel();
    let num_threads = thread::available_parallelism()?.get().min(heights.len());
    let chunk_size = (heights.len() + num_threads - 1) / num_threads;

    let mut handles = Vec::new();
    let trace_evals = trace_evaluations.0;
    let comp_evals = comp_evaluations.0;
    let weights_slice = weights.0;

    for (chunk_idx, chunk) in heights.chunks(chunk_size).enumerate() {
        let start_idx = chunk_idx * chunk_size;
        let full_widths_chunk = &full_widths[start_idx..(start_idx + chunk.len()).min(full_widths.len())];
        let omicrons_chunk = &omicrons[start_idx..(start_idx + chunk.len()).min(omicrons.len())];
        let tx = tx.clone();
        let omega_pow = omega_pow.clone();
        let deep_challenge = deep_challenge.clone();
        let new_comp_eval = new_comp_eval.clone();
        let trace_elems = trace_elems.clone();
        let trace_evals = trace_evals.to_vec();
        let comp_elems = comp_elems.clone();
        let comp_evals = comp_evals.to_vec();
        let weights_slice = weights_slice.to_vec();
        let num_comp_pieces = num_comp_pieces;

        let handle = thread::spawn(move || {
            let mut acc = Felt::zero();
            let mut num = start_idx;
            let mut total_full_width = full_widths[..start_idx].iter().sum::<u64>() as usize;

            for (j, (&_height, &omicron)) in chunk.iter().zip(omicrons_chunk).enumerate() {
                let full_width = full_widths_chunk[j] as usize;
                let current_trace_elems = &trace_elems[total_full_width..(total_full_width + full_width)];

                // Process trace evaluations (first row)
                let denom1 = fsub_(&omega_pow, &deep_challenge);
                (acc, num) = process_belt(
                    current_trace_elems,
                    &trace_evals,
                    &weights_slice,
                    full_width,
                    num,
                    &denom1,
                    &acc,
                );

                // Process trace evaluations (second row, shifted by omicron)
                let denom2 = fsub_(&omega_pow, &fmul_(&deep_challenge, omicron));
                (acc, num) = process_belt(
                    current_trace_elems,
                    &trace_evals,
                    &weights_slice,
                    full_width,
                    num,
                    &denom2,
                    &acc,
                );

                // Process new_comp_eval (first row)
                let denom3 = fsub_(&omega_pow, &new_comp_eval);
                (acc, num) = process_belt(
                    current_trace_elems,
                    &trace_evals,
                    &weights_slice,
                    full_width,
                    num,
                    &denom3,
                    &acc,
                );

                // Process new_comp_eval (second row)
                let denom4 = fsub_(&omega_pow, &fmul_(&new_comp_eval, omicron));
                (acc reward, num) = process_belt(
                    current_trace_elems,
                    &trace_evals,
                    &weights_slice,
                    full_width,
                    num,
                    &denom4,
                    &acc,
                );

                total_full_width += full_width;
            }

            // Process composition elements (only for the last chunk)
            if chunk_idx == heights.len() - 1 {
                let denom = fsub_(&omega_pow, &fpow_(&deep_challenge, num_comp_pieces));
                let (comp_acc, _) = process_belt(
                    &comp_elems,
                    &comp_evals,
                    &weights_slice[num..],
                    num_comp_pieces as usize,
                    0,
                    &denom,
                    &acc,
                );
                acc = comp_acc;
            }

            tx.send((acc, num)).unwrap();
        });
        handles.push((handle));
    }

    // Collect results
    let mut acc = Felt::zero();
    let mut num_max = 0usize;
    for _ in 0..num_threads {
        let (thread_acc, thread_num) = rx.recv().map_err(|_| jet_err())?;
        acc = fadd_(&acc, &thread_acc);
        num_max = num_max.max(thread_num);
    }
    for handle in handles {
        handle.join().unwrap();
    }

    // Return result
    let (res, res_felt) = new_handle_mut_felt(context, &mut context.stack);
    *res_felt = acc;
    Ok(res.as_noun())
}

// Helper function for processing belts
fn process_belt(
    elems: &[Belt],
    evals: &[Felt],
    weights: &[Felt],
    width: usize,
    start_num: usize,
    denom: &Felt,
    acc_start: &Felt,
) -> (Felt, usize) {
    let mut acc = *acc_start;
    let mut num = start_num;

    for i in 0..width {
        let elem_val = Felt::lift(elems[i]);
        let eval_val = evals[num];
        let weight_val = weights[num];
        let diff = fsub_(&elem_val, &eval_val);
        let term = fmul_(&fdiv_(&diff, denom), &weight_val);
        acc = fadd_(&acc, &term);
        num += 1;
    }

    (acc, num)
}

// Optimized HoonList to Vec<Belt> conversion
fn noun_to_belt_vec(context: &mut Context, noun: Noun) -> Result<Vec<Belt>, JetErr> {
    let mut result = context.stack.get_buffer::<Belt>(0); // Reuse buffer
    let mut current = noun;
    while current.is_cell() {
        let cell = current.as_cell()?;
        let value = cell.head().as_atom()?.as_u64()?;
        result.push(Belt(value));
        current = cell.tail();
    }
    if !current.is_atom() || current.as_atom()?.as_u64()? != 0 {
        return jet_err();
    }
    Ok(result)
}

// Optimized HoonList to Vec<u64> conversion
fn noun_to_u64_vec(context: &mut Context, noun: Noun) -> Result<Vec<u64>, JetErr> {
    let mut result = context.stack.get_buffer::<u64>(0); // Reuse buffer
    let mut current = noun;
    while current.is_cell() {
        let cell = current.as_cell()?;
        let value = cell.head().as_atom()?.as_u64()?;
        result.push(value);
        current = cell.tail();
    }
    if !current.is_atom() || current.as_atom()?.as_u64()? != 0 {
        return jet_err();
    }
    Ok(result)
}