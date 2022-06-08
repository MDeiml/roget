#![allow(clippy::type_complexity)]
#![allow(clippy::blocks_in_if_conditions)]
#![feature(portable_simd)]

use std::{
    borrow::Cow,
    collections::HashSet,
    num::NonZeroU8,
    simd::{Mask, Simd, ToBitMask},
};

mod solver;
pub use solver::{Rank, Solver};

include!(concat!(env!("OUT_DIR"), "/dictionary.rs"));

pub struct Wordle {
    dictionary: HashSet<&'static Word>,
}

impl Default for Wordle {
    fn default() -> Self {
        Self::new()
    }
}

impl Wordle {
    pub fn new() -> Self {
        Self {
            dictionary: HashSet::from_iter(DICTIONARY.iter().map(|(word, _)| word)),
        }
    }

    pub fn play<G: Guesser>(&self, answer: &Word, mut guesser: G) -> Option<usize> {
        let mut history = Vec::new();
        // Wordle only allows six guesses.
        // We allow more to avoid chopping off the score distribution for stats purposes.
        for i in 1..=32 {
            let guess = guesser.guess(&history);
            if guess == *answer {
                guesser.finish(i);
                return Some(i);
            }
            assert!(
                self.dictionary.contains(&guess),
                "guess '{:?}' is not in the dictionary",
                guess
            );
            let correctness = Correctness::compute(answer, &guess);
            history.push(Guess {
                word: Cow::Owned(guess),
                mask: correctness,
            });
        }
        None
    }
}

pub type Word = Simd<u8, 8>;

pub fn str_to_word(word: &str) -> Word {
    let bytes = word.as_bytes();
    Simd::from_array([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], 0, 0, 0])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Correctness {
    /// Green
    Correct,
    /// Yellow
    Misplaced,
    /// Gray
    Wrong,
}

impl Correctness {
    fn is_misplaced(letter: u8, answer: &Word, used: &mut [bool; 5]) -> bool {
        answer[0..5].iter().copied().enumerate().any(|(i, a)| {
            if a == letter && !used[i] {
                used[i] = true;
                return true;
            }
            false
        })
    }

    pub fn compute(answer: &Word, guess: &Word) -> [Self; 5] {
        let mut c = [Correctness::Wrong; 5];
        let answer_bytes = &answer[0..5];
        let guess_bytes = &guess[0..5];
        // Array indexed by lowercase ascii letters
        let mut misplaced = [0u8; (b'z' - b'a' + 1) as usize];

        // Find all correct letters
        for ((&answer, &guess), c) in answer_bytes.iter().zip(guess_bytes).zip(c.iter_mut()) {
            if answer == guess {
                *c = Correctness::Correct
            } else {
                // If the letter does not match, count it as misplaced
                misplaced[(answer - b'a') as usize] += 1;
            }
        }
        // Check all of the non matching letters if they are misplaced
        for (&guess, c) in guess_bytes.iter().zip(c.iter_mut()) {
            // If the letter was guessed wrong and the same letter was counted as misplaced
            if *c == Correctness::Wrong && misplaced[(guess - b'a') as usize] > 0 {
                *c = Correctness::Misplaced;
                misplaced[(guess - b'a') as usize] -= 1;
            }
        }

        c
    }
}

pub const MAX_MASK_ENUM: usize = 3 * 3 * 3 * 3 * 3;

/// A wrapper type for `[Correctness; 5]` packed into a single byte with a niche.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
// The NonZeroU8 here lets the compiler know that we're not using the value `0`, and that `0` can
// therefore be used to represent `None` for `Option<PackedCorrectness>`.
struct PackedCorrectness(NonZeroU8);

impl From<[Correctness; 5]> for PackedCorrectness {
    fn from(c: [Correctness; 5]) -> Self {
        let packed = c.iter().fold(0, |acc, c| {
            acc * 3
                + match c {
                    Correctness::Correct => 0,
                    Correctness::Misplaced => 1,
                    Correctness::Wrong => 2,
                }
        });
        Self(NonZeroU8::new(packed + 1).unwrap())
    }
}

impl PackedCorrectness {
    const POWERS_OF_THREE: Simd<u8, 8> = Simd::from_array([3 * 27, 27, 9, 3, 1, 0, 0, 0]);

    pub fn compute(answer: &Word, guess: &Word) -> Self {
        // This uses the negatives of correct, misplaced and used so that they don't have to be
        // negated at a later point.
        let incorrect = answer.lanes_ne(*guess);
        let mut unused = incorrect.to_bitmask();
        let mut non_existent: Mask<i8, 8> = Mask::splat(false);
        // Mask guess so that the letters that are already correct can not be marked as misplaced
        let masked_guess = incorrect.to_int().cast() & guess;
        for i in 0..5 {
            let letter: Simd<u8, 8> = Simd::splat(masked_guess[i]);
            // A letter in the is misplaced if it is equal to a letter in the answer that is not
            // used yet.
            let mask = answer.lanes_eq(letter).to_bitmask() & unused;
            non_existent.set(i as usize, mask == 0);
            // Set 6th bit to 1 to clamp index in the range [0, 5].
            // We don't care about the bits 5 to 7 of unused, so it is ok to set them.
            let index = (mask | 0b100000).trailing_zeros();
            unused &= !(1 << index);
        }
        // Use precomputed powers of 3 to quickly calculate packed representation
        let incorrects = incorrect.to_int().cast() & Self::POWERS_OF_THREE;
        let non_existents = non_existent.to_int().cast() & incorrects;
        let c = (incorrects + non_existents).reduce_sum();

        Self(NonZeroU8::new(c + 1).unwrap())
    }
}

impl From<PackedCorrectness> for u8 {
    fn from(this: PackedCorrectness) -> Self {
        this.0.get() - 1
    }
}

pub struct Guess<'a> {
    pub word: Cow<'a, Word>,
    pub mask: [Correctness; 5],
}

impl Guess<'_> {
    pub fn matches(&self, word: &Word) -> bool {
        // Check if the guess would be possible to observe when `word` is the correct answer.
        // This is equivalent to
        //     Correctness::compute(word, &self.word) == self.mask
        // without _necessarily_ computing the full mask for the tested word
        let mut used = [false; 5];

        // Check Correct letters
        for (i, (a, g)) in word[0..5]
            .iter()
            .copied()
            .zip(self.word[0..5].iter().copied())
            .enumerate()
        {
            if a == g {
                if self.mask[i] != Correctness::Correct {
                    return false;
                }
                used[i] = true;
            } else if self.mask[i] == Correctness::Correct {
                return false;
            }
        }

        // Check Misplaced letters
        for (g, e) in self.word[0..5].iter().copied().zip(self.mask.iter()) {
            if *e == Correctness::Correct {
                continue;
            }
            if Correctness::is_misplaced(g, word, &mut used) != (*e == Correctness::Misplaced) {
                return false;
            }
        }

        // The rest will be all correctly Wrong letters
        true
    }
}

pub trait Guesser {
    fn guess(&mut self, history: &[Guess]) -> Word;
    fn finish(&self, _guesses: usize) {}
}

impl Guesser for fn(history: &[Guess]) -> Word {
    fn guess(&mut self, history: &[Guess]) -> Word {
        (*self)(history)
    }
}

#[cfg(test)]
macro_rules! guesser {
    (|$history:ident| $impl:block) => {{
        struct G;
        impl $crate::Guesser for G {
            fn guess(&mut self, $history: &[Guess]) -> $crate::Word {
                $impl
            }
        }
        G
    }};
}

#[cfg(test)]
macro_rules! mask {
    (C) => {$crate::Correctness::Correct};
    (M) => {$crate::Correctness::Misplaced};
    (W) => {$crate::Correctness::Wrong};
    ($($c:tt)+) => {[
        $(mask!($c)),+
    ]}
}

#[cfg(test)]
mod tests {
    mod guess_matcher {
        use crate::str_to_word;
        use crate::Guess;
        use std::borrow::Cow;

        macro_rules! check {
            ($prev:literal + [$($mask:tt)+] allows $next:literal) => {
                assert!(Guess {
                    word: Cow::Owned(str_to_word($prev)),
                    mask: mask![$($mask )+]
                }
                .matches(&str_to_word($next)));
                assert_eq!($crate::Correctness::compute(&str_to_word($next), &str_to_word($prev)), mask![$($mask )+]);
            };
            ($prev:literal + [$($mask:tt)+] disallows $next:literal) => {
                assert!(!Guess {
                    word: Cow::Owned(str_to_word($prev)),
                    mask: mask![$($mask )+]
                }
                .matches(&str_to_word($next)));
                assert_ne!($crate::Correctness::compute(&str_to_word($next), &str_to_word($prev)), mask![$($mask )+]);
            };
        }

        #[test]
        fn from_jon() {
            check!("abcde" + [C C C C C] allows "abcde");
            check!("abcdf" + [C C C C C] disallows "abcde");
            check!("abcde" + [W W W W W] allows "fghij");
            check!("abcde" + [M M M M M] allows "eabcd");
            check!("baaaa" + [W C M W W] allows "aaccc");
            check!("baaaa" + [W C M W W] disallows "caacc");
        }

        #[test]
        fn from_crash() {
            check!("tares" + [W M M W W] disallows "brink");
        }

        #[test]
        fn from_yukosgiti() {
            check!("aaaab" + [C C C W M] allows "aaabc");
            check!("aaabc" + [C C C M W] allows "aaaab");
        }

        #[test]
        fn from_chat() {
            // flocular
            check!("aaabb" + [C M W W W] disallows "accaa");
            // ritoban
            check!("abcde" + [W W W W W] disallows "bcdea");
        }
    }
    mod game {
        use crate::{str_to_word, Guess, Wordle};

        #[test]
        fn genius() {
            let w = Wordle::new();
            let guesser = guesser!(|_history| { str_to_word("right") });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(1));
        }

        #[test]
        fn magnificent() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 1 {
                    return str_to_word("right");
                }
                str_to_word("wrong")
            });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(2));
        }

        #[test]
        fn impressive() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 2 {
                    return str_to_word("right");
                }
                str_to_word("wrong")
            });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(3));
        }

        #[test]
        fn splendid() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 3 {
                    return str_to_word("right");
                }
                str_to_word("wrong")
            });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(4));
        }

        #[test]
        fn great() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 4 {
                    return str_to_word("right");
                }
                str_to_word("wrong")
            });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(5));
        }

        #[test]
        fn phew() {
            let w = Wordle::new();
            let guesser = guesser!(|history| {
                if history.len() == 5 {
                    return str_to_word("right");
                }
                str_to_word("wrong")
            });
            assert_eq!(w.play(&str_to_word("right"), guesser), Some(6));
        }

        #[test]
        fn oops() {
            let w = Wordle::new();
            let guesser = guesser!(|_history| { str_to_word("wrong") });
            assert_eq!(w.play(&str_to_word("right"), guesser), None);
        }
    }

    mod compute {
        use crate::{str_to_word, Correctness};

        #[test]
        fn all_green() {
            assert_eq!(
                Correctness::compute(&str_to_word("abcde"), &str_to_word("abcde")),
                mask![C C C C C]
            );
        }

        #[test]
        fn all_gray() {
            assert_eq!(
                Correctness::compute(&str_to_word("abcde"), &str_to_word("fghij")),
                mask![W W W W W]
            );
        }

        #[test]
        fn all_yellow() {
            assert_eq!(
                Correctness::compute(&str_to_word("abcde"), &str_to_word("eabcd")),
                mask![M M M M M]
            );
        }

        #[test]
        fn repeat_green() {
            assert_eq!(
                Correctness::compute(&str_to_word("aabbb"), &str_to_word("aaccc")),
                mask![C C W W W]
            );
        }

        #[test]
        fn repeat_yellow() {
            assert_eq!(
                Correctness::compute(&str_to_word("aabbb"), &str_to_word("ccaac")),
                mask![W W M M W]
            );
        }

        #[test]
        fn repeat_some_green() {
            assert_eq!(
                Correctness::compute(&str_to_word("aabbb"), &str_to_word("caacc")),
                mask![W C M W W]
            );
        }

        #[test]
        fn dremann_from_chat() {
            assert_eq!(
                Correctness::compute(&str_to_word("azzaz"), &str_to_word("aaabb")),
                mask![C M W W W]
            );
        }

        #[test]
        fn itsapoque_from_chat() {
            assert_eq!(
                Correctness::compute(&str_to_word("baccc"), &str_to_word("aaddd")),
                mask![W C W W W]
            );
        }

        #[test]
        fn ricoello_from_chat() {
            assert_eq!(
                Correctness::compute(&str_to_word("abcde"), &str_to_word("aacde")),
                mask![C W C C C]
            );
        }
    }
}
