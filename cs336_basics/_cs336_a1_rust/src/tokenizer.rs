use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, prelude::*};
use std::path::PathBuf;

use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;

#[pyfunction]
pub fn write_vocab(path: PathBuf, vocab: BTreeMap<u32, Vec<u8>>) -> anyhow::Result<()> {
    let mut vf = File::create(path)?;
    for token in vocab.values() {
        writeln!(vf, "{}", String::from_utf8(escape_bytes::escape(token))?)?;
    }

    Ok(())
}

#[pyfunction]
pub fn read_vocab(path: PathBuf) -> anyhow::Result<BTreeMap<u32, Vec<u8>>> {
    let mut vf = BufReader::new(File::open(path)?);
    let mut vocab = BTreeMap::new();
    let mut line = String::new();
    while let Ok(size) = vf.read_line(&mut line) {
        if size == 0 {
            break;
        }
        let line_no_newline = line.strip_suffix('\n').unwrap_or(&line);
        let token = escape_bytes::unescape(line_no_newline.as_bytes())
            .map_err(|e| anyhow::anyhow!("unescape err {:?}", e))?;
        vocab.insert(vocab.len() as u32, token);
        line.clear();
    }
    Ok(vocab)
}

#[pyfunction]
pub fn write_merges(path: PathBuf, merges: Vec<(Vec<u8>, Vec<u8>)>) -> anyhow::Result<()> {
    let mut mf = File::create(path)?;
    for merge in merges {
        writeln!(
            mf,
            "{}\t{}",
            String::from_utf8(escape_bytes::escape(&merge.0))?,
            String::from_utf8(escape_bytes::escape(&merge.1))?,
        )?;
    }

    Ok(())
}

#[pyfunction]
pub fn read_merges(path: PathBuf) -> anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut vf = BufReader::new(File::open(path)?);
    let mut merges = Vec::new();
    let mut line = String::new();
    while let Ok(size) = vf.read_line(&mut line) {
        if size == 0 {
            break;
        }
        let line_no_newline = line.strip_suffix('\n').unwrap_or(&line);
        if let Some((left, right)) = line_no_newline.split_once('\t') {
            let left = escape_bytes::unescape(left.as_bytes())
                .map_err(|e| anyhow::anyhow!("unescape err {:?}", e))?;
            let right = escape_bytes::unescape(right.as_bytes())
                .map_err(|e| anyhow::anyhow!("unescape err {:?}", e))?;
            merges.push((left, right));
        } else {
            return Err(anyhow::anyhow!(
                "line did not have two tab separated parts {:?}",
                line_no_newline
            ));
        }
        line.clear()
    }
    Ok(merges)
}

#[pyfunction]
pub fn rust_run_train_bpe(
    input_path: PathBuf,
    vocab_size: u32,
    special_tokens: Vec<String>,
) -> anyhow::Result<(BTreeMap<u32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
    dbg!(&input_path);
    dbg!(&vocab_size);
    dbg!(&special_tokens);

    let special_tokens_re = Regex::new(
        &special_tokens
            .iter()
            .map(|s| regex::escape(s))
            .collect::<Vec<_>>()
            .join("|"),
    )?;

    let entire_corpus = std::fs::read_to_string(input_path)?;
    let corpus_chunks = special_tokens_re
        .split(&entire_corpus)
        .map(String::from)
        .collect::<Vec<String>>();
    println!("corpus_chunks {}", corpus_chunks.len());
    std::mem::drop(entire_corpus);

    // Run pre-tokenization and counting of chunks in parallel
    let pretok_re = gpt2_pretok_re();
    let partial_counts = corpus_chunks
        .into_par_iter()
        .fold(
            || SequenceTracker::default(),
            |mut tracker: SequenceTracker, chunk: String| {
                let new = SequenceTracker::from_chunk(&pretok_re, &chunk);
                tracker.join_with(new);
                tracker
            },
        )
        .collect_vec_list();

    // Fold into a single set of counts
    let mut counts = SequenceTracker::default();
    for chunk_counts in partial_counts.into_iter().flat_map(|v| v.into_iter()) {
        counts.join_with(chunk_counts);
    }
    println!("counts len {}", counts.0.len());

    let mut vocab = BTreeMap::default();
    for special_token in special_tokens {
        vocab.insert(vocab.len() as u32, special_token.into_bytes());
    }
    for byte in 0..256 {
        vocab.insert(vocab.len() as u32, vec![byte as u8]);
    }

    let mut merge_list: Vec<(Vec<u8>, Vec<u8>)> = Default::default();
    while vocab.len() < vocab_size as usize {
        let merge = counts.find_next_merge();
        let merged = merge_vecs(merge.0.clone(), merge.1.clone());
        vocab.insert(vocab.len() as u32, merged);
        merge_list.push(merge.clone());
        counts.apply_merge(merge);
    }

    Ok((vocab, merge_list))
}

#[derive(Default, Clone, Debug)]
struct SequenceTracker(pub HashMap<Vec<Vec<u8>>, u64>);

impl SequenceTracker {
    /// Joins two sequence trackers (from two different text chunks) together
    pub fn join_with(&mut self, other: SequenceTracker) {
        for (k, count) in other.0.into_iter() {
            *self.0.entry(k).or_insert(0) += count;
        }
    }

    pub fn from_chunk(pretok_re: &fancy_regex::Regex, chunk: &str) -> Self {
        let mut tracker = HashMap::default();
        for m in pretok_re.find_iter(chunk) {
            let k = m
                .unwrap()
                .as_str()
                .as_bytes()
                .iter()
                .map(|b| vec![*b])
                .collect::<Vec<Vec<u8>>>();

            *tracker.entry(k).or_default() += 1;
        }

        Self(tracker)
    }

    pub fn pair_frequencies(&self) -> BTreeMap<(&Vec<u8>, &Vec<u8>), u64> {
        let mut freqs = BTreeMap::default();

        for (k, count) in self.0.iter() {
            let lefts = k.iter();
            let rights = k.iter().skip(1);
            for (left, right) in lefts.zip(rights) {
                *freqs.entry((left, right)).or_default() += count;
            }
        }

        freqs
    }

    pub fn find_next_merge(&self) -> (Vec<u8>, Vec<u8>) {
        let pair_freqs = self.pair_frequencies();

        let (mut highest_pair, mut highest_score) = pair_freqs.iter().next().unwrap();
        for (pair, score) in &pair_freqs {
            if score > highest_score {
                highest_pair = pair;
                highest_score = score;
            } else if score == highest_score && pair != highest_pair {
                // horrible, no good, replicating python nested lex order semantics
                // kind of nasty but matches semantics of test cases exactly
                if pair
                    .0
                    .cmp(highest_pair.0)
                    .then_with(|| pair.1.cmp(highest_pair.1))
                    == std::cmp::Ordering::Greater
                {
                    highest_pair = pair;
                    highest_score = score;
                }
            }
        }

        (highest_pair.0.clone(), highest_pair.1.clone())
    }

    pub fn apply_merge(&mut self, merge: (Vec<u8>, Vec<u8>)) {
        let (left_merge, right_merge) = merge;
        let merged = merge_vecs(left_merge.clone(), right_merge.clone());

        let old_tracker = std::mem::replace(&mut self.0, Default::default());

        for (seq, count) in old_tracker {
            // In this case, we don't need to touch the item (fast path)
            if !seq.contains(&left_merge) || !seq.contains(&right_merge) {
                *self.0.entry(seq).or_default() += count;
            } else {
                // Otherwise, we have both parts of the pair, try to apply a merge
                let placeholder = Vec::new();
                let lookahead_seq = seq.iter().skip(1).chain(std::iter::once(&placeholder));

                let mut seq_with_lookahead = seq.iter().zip(lookahead_seq);
                let mut new_seq = Vec::with_capacity(seq.len());
                while let Some((current, lookahead)) = seq_with_lookahead.next() {
                    if (current, lookahead) == (&left_merge, &right_merge) {
                        new_seq.push(merged.clone());
                        // Skip the second half of the pair
                        let _ = seq_with_lookahead.next();
                    } else {
                        new_seq.push(current.clone());
                    }
                }

                *self.0.entry(new_seq).or_default() += count;
            }
        }
    }
}

#[allow(unused)]
fn debug_merge(m: (&Vec<u8>, &Vec<u8>)) -> String {
    let left_s = String::from_utf8_lossy(m.0);
    let right_s = String::from_utf8_lossy(m.1);
    format!(
        "{} {} [{:?} {:?}]",
        left_s.escape_debug(),
        right_s.escape_debug(),
        m.0,
        m.1
    )
}

fn merge_vecs(mut a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    a.extend_from_slice(&b);
    a
}

fn gpt2_pretok_re() -> fancy_regex::Regex {
    fancy_regex::Regex::new(
        r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretok_with_re() {
        let input = "some text that i'll pre-tokenize";
        let matches = gpt2_pretok_re()
            .find_iter(&input)
            .flat_map(|m| m.ok().map(|m| m.as_str().to_owned()))
            .collect::<Vec<String>>();
        assert_eq!(
            matches,
            vec![
                "some", " text", " that", " i", "'ll", " pre", "-", "tokenize"
            ]
        );
    }

    #[test]
    fn handout_example() {
        let corpus = r#"low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"#;
        let re = fancy_regex::Regex::new(r"\S+").unwrap();
        let tracker = SequenceTracker::from_chunk(&re, &corpus);
        println!("{:?}", tracker);
        let pair_freqs = tracker.pair_frequencies();
        println!("{:?}", pair_freqs);

        let expected_freqs: BTreeMap<(Vec<u8>, Vec<u8>), u64> = [
            ((vec![108], vec![111]), 7),
            ((vec![111], vec![119]), 7),
            ((vec![119], vec![101]), 8),
            ((vec![101], vec![114]), 2),
            ((vec![119], vec![105]), 3),
            ((vec![105], vec![100]), 3),
            ((vec![100], vec![101]), 3),
            ((vec![101], vec![115]), 9),
            ((vec![115], vec![116]), 9),
            ((vec![110], vec![101]), 6),
            ((vec![101], vec![119]), 6),
        ]
        .iter()
        .cloned()
        .collect();
        let expected_freqs_borrow = expected_freqs
            .iter()
            .map(|(k, v)| ((&k.0, &k.1), *v))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(pair_freqs, expected_freqs_borrow);

        let first_merge = tracker.find_next_merge();

        assert_eq!(first_merge.0, vec![115]); // s
        assert_eq!(first_merge.1, vec![116]); // t

        let debug_merge = |(a, b)| (String::from_utf8(a).unwrap(), String::from_utf8(b).unwrap());

        let mut tracker2 = tracker.clone();
        tracker2.apply_merge(first_merge);
        println!("tracker2 {:?}", tracker2);

        let mut tracker3 = tracker2.clone();
        let merge2 = tracker2.find_next_merge();
        println!("merge2 {:?}", debug_merge(merge2.clone()));
        tracker3.apply_merge(merge2);
        println!("tracker3 {:?}", tracker3);

        let mut tracker4 = tracker3.clone();
        let merge3 = tracker3.find_next_merge();
        println!("merge3 {:?}", debug_merge(merge3.clone()));
        tracker4.apply_merge(merge3);
        println!("tracker4 {:?}", tracker4);

        let mut tracker5 = tracker4.clone();
        let merge4 = tracker4.find_next_merge();
        println!("merge4 {:?}", debug_merge(merge4.clone()));
        tracker5.apply_merge(merge4);
        println!("tracker5 {:?}", tracker5);

        let mut tracker6 = tracker5.clone();
        let merge5 = tracker5.find_next_merge();
        println!("merge5 {:?}", debug_merge(merge5.clone()));
        tracker6.apply_merge(merge5);
        println!("tracker6 {:?}", tracker6);
    }

    #[test]
    fn test_train_bpe() -> anyhow::Result<()> {
        let input_path = PathBuf::from(
            "/Users/hdevalence/code/stanford-cs336/cs336-assignment1-basics/tests/fixtures/corpus.en",
        );
        let vocab_size = 300;
        let special_tokens = vec!["<|endoftext|>".to_string()];

        let (vocab, merges) = rust_run_train_bpe(input_path, vocab_size, special_tokens)?;

        write_vocab("test-fixtures-vocab.txt".into(), vocab.clone())?;
        write_merges("test-fixtures-merges.txt".into(), merges.clone())?;

        let vocab2 = read_vocab("test-fixtures-vocab.txt".into())?;
        let merges2 = read_merges("test-fixtures-merges.txt".into())?;

        assert_eq!(vocab, vocab2);
        assert_eq!(merges, merges2);

        Ok(())
    }

    #[test]
    fn test_train_tinystories() -> anyhow::Result<()> {
        let input_path = PathBuf::from(
            "/Users/hdevalence/code/stanford-cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        );
        let vocab_size = 10_000;
        let special_tokens = vec!["<|endoftext|>".to_string()];

        let (vocab, merges) = rust_run_train_bpe(input_path, vocab_size, special_tokens).unwrap();

        write_vocab("test-tinystories-vocab.txt".into(), vocab.clone())?;
        write_merges("test-tinystories-merges.txt".into(), merges.clone())?;

        let vocab2 = read_vocab("test-tinystories-vocab.txt".into())?;
        let merges2 = read_merges("test-tinystories-merges.txt".into())?;

        assert_eq!(vocab, vocab2);
        assert_eq!(merges, merges2);

        Ok(())
    }

    #[test]
    fn test_train_owt() -> anyhow::Result<()> {
        let input_path = PathBuf::from(
            "/Users/hdevalence/code/stanford-cs336/cs336-assignment1-basics/data/owt_train.txt",
        );
        let vocab_size = 30_000;
        let special_tokens = vec!["<|endoftext|>".to_string()];

        let (vocab, merges) = rust_run_train_bpe(input_path, vocab_size, special_tokens).unwrap();

        write_vocab("test-owt-vocab.txt".into(), vocab.clone())?;
        write_merges("test-owt-merges.txt".into(), merges.clone())?;

        let vocab2 = read_vocab("test-owt-vocab.txt".into())?;
        let merges2 = read_merges("test-owt-merges.txt".into())?;

        assert_eq!(vocab, vocab2);
        assert_eq!(merges, merges2);

        Ok(())
    }
}
