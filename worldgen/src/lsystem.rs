use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const MAX_LSYSTEM_BYTES: usize = 1 << 20;

pub type WeightedList<T> = Vec<(f64, T)>;

#[derive(Debug, Clone)]
pub enum Production<Symbol> {
    /// Produces a fixed list of symbols.
    Fixed(Vec<Symbol>),
    /// Selects a production randomly with weights, then applies that production.
    Randomized(WeightedList<Box<Production<Symbol>>>),
}

pub type Productions<Symbol> = HashMap<Symbol, Production<Symbol>>;

#[derive(Debug, Clone)]
pub struct LSystem<Symbol> {
    pub axiom: Vec<Symbol>,
    pub productions: Productions<Symbol>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LSystemConfig<Symbol> {
    pub symbol_map: HashMap<char, Symbol>,
    pub axiom: String,
    pub productions: HashMap<char, Vec<String>>,
}

impl<Symbol> LSystemConfig<Symbol> {
    pub fn as_l_system(&self) -> Result<LSystem<Symbol>>
    where
        Symbol: Clone + Eq + std::hash::Hash,
    {
        let char_to_symbol = |c: char| -> Result<Symbol> {
            self.symbol_map
                .get(&c)
                .ok_or_else(|| anyhow!("Symbol not defined: {:?}", c))
                .map(Symbol::clone)
        };

        let str_to_symbols =
            |s: &str| -> Result<Vec<Symbol>> { s.chars().map(char_to_symbol).collect() };

        let axiom: Vec<Symbol> = str_to_symbols(&self.axiom[..])?;

        let productions: Productions<Symbol> = self
            .productions
            .iter()
            .map(|(&c, choices)| {
                let source = char_to_symbol(c)?;
                let production = if choices.len() == 1 {
                    Production::Fixed(str_to_symbols(&choices[0][..])?)
                } else {
                    Production::Randomized(
                        choices
                            .iter()
                            .map(|s: &String| {
                                let symbols = str_to_symbols(&s[..])?;
                                Ok((1.0, Box::new(Production::Fixed(symbols))))
                            })
                            .collect::<Result<_>>()?,
                    )
                };

                Ok((source, production))
            })
            .collect::<Result<_>>()?;

        Ok(LSystem { axiom, productions })
    }
}

impl<Symbol> LSystem<Symbol> {
    pub fn run<R>(&self, steps: i32, rng: &mut R) -> Result<Vec<Symbol>>
    where
        R: Rng,
        Symbol: Clone + Eq + std::hash::Hash,
    {
        let mut runner = LSystemRunner {
            description: self,
            state: self.axiom.clone(),
            rng: rng,
        };

        for i in 0..steps {
            runner
                .step()
                .with_context(|| format!("L system failed after {} steps", i))?;
        }

        Ok(runner.state)
    }

    fn produce<R>(&self, symbol: Symbol, rng: &mut R, result: &mut Vec<Symbol>)
    where
        R: Rng,
        Symbol: Clone + std::hash::Hash + Eq,
    {
        let mut production = match self.productions.get(&symbol) {
            None => {
                result.push(symbol);
                return;
            }
            Some(rule) => rule,
        };

        loop {
            match production {
                Production::Fixed(symbols) => {
                    result.extend_from_slice(&symbols[..]);
                    break;
                }
                Production::Randomized(choices) => {
                    let distribution =
                        WeightedIndex::new(choices.iter().map(|(weight, _)| weight))
                            .expect("bad weights");

                    let index = distribution.sample(rng);
                    production = &choices[index].1;
                }
            }
        }
    }
}

struct LSystemRunner<'a, Symbol, Rng> {
    description: &'a LSystem<Symbol>,
    state: Vec<Symbol>,
    rng: &'a mut Rng,
}

impl<'a, Symbol, R> LSystemRunner<'a, Symbol, R>
where
    Symbol: Clone + Eq + std::hash::Hash,
    R: Rng,
{
    fn step(&mut self) -> Result<()> {
        let max_len = MAX_LSYSTEM_BYTES / std::mem::size_of::<Symbol>();

        let mut prev_state = Vec::new();
        std::mem::swap(&mut self.state, &mut prev_state);
        for symbol in prev_state.into_iter() {
            self.description
                .produce(symbol, &mut self.rng, &mut self.state);

            if self.state.len() > max_len {
                bail!("L system length exceeded max of {}", max_len);
            }
        }

        Ok(())
    }
}
