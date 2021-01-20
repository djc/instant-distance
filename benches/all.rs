use bencher::{benchmark_group, benchmark_main, Bencher};
use rand::rngs::{StdRng, ThreadRng};
use rand::{Rng, SeedableRng};

use instant_distance::Builder;

benchmark_main!(benches);
benchmark_group!(benches, build_heuristic);

fn build_heuristic(bench: &mut Bencher) {
    let seed = ThreadRng::default().gen::<u64>();
    let mut rng = StdRng::seed_from_u64(seed);
    let points = (0..16384)
        .into_iter()
        .map(|_| Point(rng.gen(), rng.gen()))
        .collect::<Vec<_>>();

    bench.iter(|| Builder::default().seed(seed).build(&points))
}

/*
fn randomized(builder: Builder) -> (u64, usize) {
    let query = Point(rng.gen(), rng.gen());
    let mut nearest = Vec::with_capacity(256);
    for (i, p) in points.iter().enumerate() {
        nearest.push((OrderedFloat::from(query.distance(p)), i));
        if nearest.len() >= 200 {
            nearest.sort_unstable();
            nearest.truncate(100);
        }
    }

    let mut search = Search::default();
    let mut results = vec![PointId::default(); 100];
    let found = hnsw.search(&query, &mut results, &mut search);
    assert_eq!(found, 100);

    nearest.sort_unstable();
    nearest.truncate(100);
    let forced = nearest
        .iter()
        .map(|(_, i)| pids[*i])
        .collect::<HashSet<_>>();
    let found = results.into_iter().take(found).collect::<HashSet<_>>();
    (seed, forced.intersection(&found).count())
}
*/

#[derive(Clone, Copy, Debug)]
struct Point(f32, f32);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
    }
}
