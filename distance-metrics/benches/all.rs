use bencher::{benchmark_group, benchmark_main, Bencher};

use distance_metrics::{
    Metric, {CosineMetric, EuclidMetric},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

benchmark_main!(benches);
benchmark_group!(
    benches,
    legacy,
    non_simd,
    metric::<EuclidMetric>,
    metric::<CosineMetric>
);

fn legacy(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let point_a = distance_metrics::FloatArray([rng.gen(); 300]);
    let point_b = distance_metrics::FloatArray([rng.gen(); 300]);

    bench.iter(|| distance_metrics::legacy_distance(&point_a, &point_b))
}

fn non_simd(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let point_a = [rng.gen(); 300];
    let point_b = [rng.gen(); 300];

    bench.iter(|| distance_metrics::euclid_distance(&point_a, &point_b))
}

fn metric<M: Metric>(bench: &mut Bencher) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut point_a = [rng.gen(); 300];
    let mut point_b = [rng.gen(); 300];
    M::preprocess(&mut point_a);
    M::preprocess(&mut point_b);

    bench.iter(|| M::distance(&point_a, &point_b))
}

const SEED: u64 = 123456789;
