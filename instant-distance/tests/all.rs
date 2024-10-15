use std::collections::HashSet;

use ordered_float::OrderedFloat;
use rand::rngs::{StdRng, ThreadRng};
use rand::{Rng, SeedableRng};

use instant_distance::{Builder, Point as _, Search};

#[test]
#[allow(clippy::float_cmp, clippy::approx_constant)]
fn map() {
    let points = (0..5)
        .map(|i| Point(i as f32, i as f32))
        .collect::<Vec<_>>();
    let values = vec!["zero", "one", "two", "three", "four"];

    let seed = ThreadRng::default().gen::<u64>();
    println!("map (seed = {seed})");
    let map = Builder::default().seed(seed).build(points, values);
    let mut search = Search::default();

    for (i, item) in map.search(&Point(2.0, 2.0), &mut search).enumerate() {
        match i {
            0 => {
                assert_eq!(item.distance, 0.0);
                assert_eq!(item.value, &"two");
            }
            1 | 2 => {
                assert_eq!(item.distance, 1.4142135);
                assert!(item.value == &"one" || item.value == &"three");
            }
            3 | 4 => {
                assert_eq!(item.distance, 2.828427);
                assert!(item.value == &"zero" || item.value == &"four");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn random_heuristic() {
    let (seed, recall) = randomized(Builder::default());
    println!("heuristic (seed = {seed}) recall = {recall}");
    assert!(recall > 97, "expected at least 98, got {recall}");
}

#[test]
fn random_simple() {
    let (seed, recall) = randomized(Builder::default().select_heuristic(None));
    println!("simple (seed = {seed}) recall = {recall}");
    assert!(recall > 90, "expected at least 90, got {recall}");
}

fn randomized(builder: Builder) -> (u64, usize) {
    let seed = ThreadRng::default().gen::<u64>();
    let mut rng = StdRng::seed_from_u64(seed);
    let points = (0..1024)
        .map(|_| Point(rng.gen(), rng.gen()))
        .collect::<Vec<_>>();

    let query = Point(rng.gen(), rng.gen());
    let mut nearest = Vec::with_capacity(256);
    for (i, p) in points.iter().enumerate() {
        nearest.push((OrderedFloat::from(query.distance(p)), i));
        if nearest.len() >= 200 {
            nearest.sort_unstable();
            nearest.truncate(100);
        }
    }

    let (hnsw, pids) = builder.seed(seed).build_hnsw(points);
    let mut search = Search::default();
    let results = hnsw.search(&query, &mut search);
    assert!(results.len() >= 100);

    nearest.sort_unstable();
    nearest.truncate(100);
    let forced = nearest
        .iter()
        .map(|(_, i)| pids[*i])
        .collect::<HashSet<_>>();
    let found = results
        .take(100)
        .map(|item| item.pid)
        .collect::<HashSet<_>>();
    (seed, forced.intersection(&found).count())
}

#[derive(Clone, Copy, Debug)]
struct Point(f32, f32);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
    }
}

#[test]
#[allow(clippy::float_cmp, clippy::approx_constant)]
fn incremental_insert() {
    let points = (0..4)
        .map(|i| Point(i as f32, i as f32))
        .collect::<Vec<_>>();
    let values = vec!["zero", "one", "two", "three"];
    let seed = ThreadRng::default().gen::<u64>();
    let builder = Builder::default().seed(seed);

    let mut map = builder.build(points, values);

    map.insert(Point(4.0, 4.0), "four").expect("Should insert");

    let mut search = Search::default();

    for (i, item) in map.search(&Point(4.0, 4.0), &mut search).enumerate() {
        match i {
            0 => {
                assert_eq!(item.distance, 0.0);
                assert_eq!(item.value, &"four");
            }
            1 => {
                assert_eq!(item.distance, 1.4142135);
                assert!(item.value == &"three");
            }
            2 => {
                assert_eq!(item.distance, 2.828427);
                assert!(item.value == &"two");
            }
            3 => {
                assert_eq!(item.distance, 4.2426405);
                assert!(item.value == &"one");
            }
            4 => {
                assert_eq!(item.distance, 5.656854);
                assert!(item.value == &"zero");
            }
            _ => unreachable!(),
        }
    }

    // Note 
    // This has the same expected results as incremental_insert but builds
    // the whole map in one go. Only here for comparison.
    {
        let points = (0..5)
            .map(|i| Point(i as f32, i as f32))
            .collect::<Vec<_>>();
        let values = vec!["zero", "one", "two", "three", "four"];
        let seed = ThreadRng::default().gen::<u64>();
        let builder = Builder::default().seed(seed);
        let map = builder.build(points, values);
        let mut search = Search::default();
        for (i, item) in map.search(&Point(4.0, 4.0), &mut search).enumerate() {
            match i {
                0 => {
                    assert_eq!(item.distance, 0.0);
                    assert_eq!(item.value, &"four");
                }
                1 => {
                    assert_eq!(item.distance, 1.4142135);
                    assert!(item.value == &"three");
                }
                2 => {
                    assert_eq!(item.distance, 2.828427);
                    assert!(item.value == &"two");
                }
                3 => {
                    assert_eq!(item.distance, 4.2426405);
                    assert!(item.value == &"one");
                }
                4 => {
                    assert_eq!(item.distance, 5.656854);
                    assert!(item.value == &"zero");
                }
                _ => unreachable!(),
            }
        }
    }
}
