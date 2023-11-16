use aligned_vec::avec;
use bencher::{benchmark_group, benchmark_main, Bencher};

use instant_distance::{Builder, Metric, Search};
use instant_distance_py::{EuclidMetric, PointStorage};
use rand::{rngs::StdRng, Rng, SeedableRng};

benchmark_main!(benches);
benchmark_group!(benches, distance, build, query);

fn distance(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let point_a = avec![rng.gen(); 304];
    let point_b = avec![rng.gen(); 304];

    bench.iter(|| EuclidMetric::distance(&point_a, &point_b));
}

fn build(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points = (0..1024)
        .map(|_| vec![rng.gen(); 304])
        .collect::<Vec<_>>();

    bench.iter(|| {
        Builder::default()
            .seed(SEED)
            .build_hnsw::<Vec<f32>, [f32], EuclidMetric, PointStorage>(points.clone())
    });
}

fn query(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points = (0..1024)
        .map(|_| vec![rng.gen(); 304])
        .collect::<Vec<_>>();
    let (hnsw, _) = Builder::default()
        .seed(SEED)
        .build_hnsw::<Vec<f32>, [f32], EuclidMetric, PointStorage>(points);
    let point = avec![rng.gen(); 304];

    bench.iter(|| {
        let mut search = Search::default();
        let _ = hnsw.search(&point, &mut search);
    });
}

const SEED: u64 = 123456789;
