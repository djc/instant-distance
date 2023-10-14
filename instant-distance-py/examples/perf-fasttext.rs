//! This is a simple example of how to use the instant-distance crate to build an index from a file.
//! The file is expected to be in the format of the fasttext word vectors.
//!
//! This example was built to performance using such tools as valgrind, perf, and heaptrack.
//!
//! Get the required file:
//! wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use clap::Parser;
use instant_distance::{Builder, Search};

use instant_distance_py::FloatArray;

fn main() -> Result<(), anyhow::Error> {
    let opt = Options::parse();
    let (words, points) = {
        let path: &Path = &opt.path;
        let count = opt.count;
        let mut words = vec![];
        let mut points = vec![];
        let mut reader = BufReader::new(File::open(path)?);

        // skip first line
        let mut discarded_line = String::new();
        let _bytes_read = reader.read_line(&mut discarded_line)?;

        for _ in 0..count {
            let mut line = String::new();
            let _read_bytes = reader.read_line(&mut line)?;
            let mut parts = line.split(' ');
            let word = parts.next().unwrap();
            words.push(word.to_string());
            let rest = parts
                .flat_map(|s| s.trim().parse::<f32>().ok())
                .collect::<Vec<_>>();

            let mut float_array_inner = [0f32; 300];
            float_array_inner.copy_from_slice(&rest[..300]);
            let float_array = FloatArray(float_array_inner);

            points.push(float_array);
        }

        (words, points)
    };
    println!(
        "{} points loaded, building hnsw(seed: {})...",
        points.len(),
        opt.seed
    );

    let wait = opt.wait;
    let num_queries = opt.num_queries;
    let points = points.clone();
    let start = Instant::now();
    let hnswmap = Builder::default().seed(opt.seed).build(points, words);
    println!("contiguous indexing took {:?}", start.elapsed());

    if wait {
        println!("sleeping for 15s");
        sleep(Duration::from_millis(15000));
    }

    let mut search = Search::default();
    let point = FloatArray([0.0; 300]);
    for _ in 0..20 {
        let query_start = Instant::now();
        for _ in 0..num_queries {
            let _closest_point = hnswmap.search(&point, &mut search).next().unwrap();
            tracy_full::frame!("contiguous search");
        }

        tracy_full::frame!("search group");
        println!("{} queries took {:?}", num_queries, query_start.elapsed());
    }
    tracy_full::frame!();

    Ok(())
}

#[derive(Parser)]
struct Options {
    #[arg(
        short,
        default_value = "../instant-distance-benchmarking/wiki.en.align.vec"
    )]
    path: PathBuf,
    #[arg(short, default_value = "50000")]
    count: usize,

    #[arg(short, default_value = "1000")]
    num_queries: usize,

    #[arg(short)]
    wait: bool,

    #[structopt(short, default_value = "123456789")]
    seed: u64,
}

#[global_allocator]
static ALLOC: tracy_full::alloc::GlobalAllocator = tracy_full::alloc::GlobalAllocator::new();
