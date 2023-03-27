use bencher::{benchmark_group, benchmark_main, Bencher};

use instant_distance::{Builder, Metric, Search};
use instant_distance_py::{EuclidMetric, FloatArray};
use rand::{rngs::StdRng, Rng, SeedableRng};

benchmark_main!(benches);
benchmark_group!(benches, distance, build, query);

fn distance(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let point_a = FloatArray([rng.gen(); 300]);
    let point_b = FloatArray([rng.gen(); 300]);

    bench.iter(|| EuclidMetric::distance(&point_a, &point_b));
}

fn build(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points = (0..1024)
        .map(|_| FloatArray([rng.gen(); 300]))
        .collect::<Vec<_>>();

    bench.iter(|| {
        Builder::default()
            .seed(SEED)
            .build_hnsw::<_, _, EuclidMetric, Vec<FloatArray>>(points.clone())
    });
}

fn query(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let points = (0..1024)
        .map(|_| FloatArray([rng.gen(); 300]))
        .collect::<Vec<_>>();
    let (hnsw, _) = Builder::default()
        .seed(SEED)
        .build_hnsw::<_, _, EuclidMetric, Vec<FloatArray>>(points);
    let point = FloatArray([rng.gen(); 300]);

    bench.iter(|| {
        let mut search = Search::default();
        let _ = hnsw.search(&point, &mut search);
    });
}

const SEED: u64 = 123456789;
