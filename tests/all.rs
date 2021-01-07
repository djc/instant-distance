use ordered_float::OrderedFloat;
use rand::rngs::{StdRng, ThreadRng};
use rand::{Rng, SeedableRng};

use instant_distance::{Hnsw, Point as _, PointId, Search};

#[test]
fn basic() {
    let (hnsw, pids) = Hnsw::<Point>::builder().build(&[
        Point(0.1, 0.4),
        Point(-0.324, 0.543),
        Point(0.87, -0.33),
        Point(0.452, 0.932),
    ]);

    let mut search = Search::default();
    let mut results = vec![PointId::default()];
    let p = Point(0.1, 0.35);
    let found = hnsw.search(&p, &mut results, &mut search);
    assert_eq!(found, 1);
    assert_eq!(&results, &[pids[0]]);
}

#[test]
fn randomized() {
    let seed = ThreadRng::default().gen::<u64>();
    println!("seed {}", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let points = (0..1024)
        .into_iter()
        .map(|_| Point(rng.gen(), rng.gen()))
        .collect::<Vec<_>>();

    let query = Point(rng.gen(), rng.gen());
    println!("query: {:?}", query);

    for (i, p) in points.iter().enumerate() {
        println!("{:2} {:?} ({})", i, p, query.distance(p));
    }

    let (hnsw, pids) = Hnsw::<Point>::builder().seed(seed).build(&points);
    let mut search = Search::default();
    let mut results = vec![PointId::default()];
    let found = hnsw.search(&query, &mut results, &mut search);
    assert_eq!(found, 1);

    let nearest = points
        .iter()
        .enumerate()
        .map(|(i, other)| (OrderedFloat::from(query.distance(other)), i))
        .min()
        .unwrap();
    println!(
        "nearest (brute force): {:?} @ {:9.7}",
        pids[nearest.1],
        nearest.0.into_inner()
    );

    let index = pids.iter().position(|p| p == &results[0]).unwrap();
    println!(
        "nearest (hnsw):        {:?} @ {:9.7}",
        results[0],
        query.distance(&points[index])
    );
    assert_eq!(pids[nearest.1], results[0]);
}

#[derive(Clone, Copy, Debug)]
struct Point(f32, f32);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
    }
}
