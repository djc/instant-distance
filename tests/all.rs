use instant_distance::{Hnsw, PointId, Search};

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

#[derive(Clone, Copy, Debug)]
struct Point(f32, f32);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
    }
}
