//! Token sampling.  Reference semantics follow the model's
//! `generation_config.json`: `temperature`, `top_k`, `top_p` applied in that
//! order on the full-vocabulary logits (248320 entries for Qwen3.8).

use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Deterministic greedy decoding (used by the correctness gate).
    pub greedy: bool,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            greedy: false,
            seed: 0x5eed,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            greedy: true,
            ..Default::default()
        }
    }
}

/// Deterministic xorshift RNG so that runs are reproducible bit-for-bit.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct Sampler {
    rng: Rng,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(0x5eed)
    }
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }

    /// Sample one token id from `logits`.
    pub fn sample(&mut self, logits: &[f32], p: &SamplingParams) -> Result<u32> {
        if logits.is_empty() {
            bail!("empty logits");
        }
        if p.greedy || p.temperature <= 0.0 {
            let mut best = 0usize;
            for (i, v) in logits.iter().enumerate() {
                if *v > logits[best] {
                    best = i;
                }
            }
            return Ok(best as u32);
        }

        // temperature
        let inv_t = 1.0 / p.temperature;
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let mut vals: Vec<f32> = logits.iter().map(|v| v * inv_t).collect();

        // top-k
        if p.top_k > 0 && p.top_k < vals.len() {
            idx.sort_unstable_by(|a, b| vals[*b as usize].total_cmp(&vals[*a as usize]));
            idx.truncate(p.top_k);
            vals = idx.iter().map(|i| logits[*i as usize] * inv_t).collect();
        }

        // softmax over the surviving candidates
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = vals.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Ok(idx[0]);
        }
        for p_ in probs.iter_mut() {
            *p_ /= sum;
        }

        // top-p (nucleus): keep the smallest prefix whose mass exceeds top_p
        if p.top_p > 0.0 && p.top_p < 1.0 {
            let mut order: Vec<usize> = (0..probs.len()).collect();
            order.sort_unstable_by(|a, b| probs[*b].total_cmp(&probs[*a]));
            let mut acc = 0.0f32;
            let mut keep = Vec::new();
            for &i in &order {
                acc += probs[i];
                keep.push(i);
                if acc >= p.top_p {
                    break;
                }
            }
            let kept_sum: f32 = keep.iter().map(|i| probs[*i]).sum();
            let mut new_idx = Vec::with_capacity(keep.len());
            let mut new_p = Vec::with_capacity(keep.len());
            for &i in &keep {
                new_idx.push(idx[i]);
                new_p.push(probs[i] / kept_sum);
            }
            idx = new_idx;
            probs = new_p;
        }

        // inverse-CDF draw
        let r = self.rng.next_f32();
        let mut acc = 0.0f32;
        for (i, pr) in probs.iter().enumerate() {
            acc += *pr;
            if r < acc {
                return Ok(idx[i]);
            }
        }
        Ok(*idx.last().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::default();
        let logits = vec![0.1, 5.0, -3.0, 4.9];
        assert_eq!(s.sample(&logits, &SamplingParams::greedy()).unwrap(), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(7);
        let logits = vec![0.1, 5.0, -3.0, 4.9];
        let p = SamplingParams {
            top_k: 1,
            greedy: false,
            ..Default::default()
        };
        assert_eq!(s.sample(&logits, &p).unwrap(), 1);
    }

    #[test]
    fn sampling_is_reproducible_with_a_seed() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let p = SamplingParams {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.9,
            greedy: false,
            seed: 42,
        };
        let mut a = Sampler::new(42);
        let mut b = Sampler::new(42);
        let sa: Vec<u32> = (0..32).map(|_| a.sample(&logits, &p).unwrap()).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.sample(&logits, &p).unwrap()).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut s = Sampler::new(1);
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let logits = vec![1.0, 9.0, 2.0];
        assert_eq!(s.sample(&logits, &p).unwrap(), 1);
    }
}
