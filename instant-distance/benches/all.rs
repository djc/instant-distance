use bencher::{benchmark_group, benchmark_main, Bencher};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use instant_distance::{Builder, Metric};

benchmark_main!(benches);
benchmark_group!(benches, build_heuristic);

fn build_heuristic(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points = (0..1024)
        .map(|_| [rng.gen(), rng.gen()])
        .collect::<Vec<_>>();

    bench.iter(|| {
        Builder::default()
            .seed(SEED)
            .build_hnsw::<[f32; 2], [f32; 2], EuclidMetric, Vec<[f32; 2]>>(points.clone())
    })
}

const SEED: u64 = 123456789;

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

struct EuclidMetric;

impl Metric<[f32; 2]> for EuclidMetric {
    fn distance(a: &[f32; 2], b: &[f32; 2]) -> f32 {
        // Euclidean distance metric
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
