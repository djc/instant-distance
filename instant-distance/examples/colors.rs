use instant_distance::{Builder, Metric, Search};

fn main() {
    let points = vec![Point(255, 0, 0), Point(0, 255, 0), Point(0, 0, 255)];
    let values = vec!["red", "green", "blue"];

    let map =
        Builder::default().build::<Point, Point, EuclidMetric, &str, Vec<Point>>(points, values);
    let mut search = Search::default();

    let burnt_orange = Point(204, 85, 0);

    let closest_point = map.search(&burnt_orange, &mut search).next().unwrap();

    println!("{:?}", closest_point.value);
}

#[derive(Clone, Copy, Debug)]
struct Point(isize, isize, isize);

struct EuclidMetric;

impl Metric<Point> for EuclidMetric {
    fn distance(a: &Point, b: &Point) -> f32 {
        // Euclidean distance metric
        (((a.0 - b.0).pow(2) + (a.1 - b.1).pow(2) + (a.2 - b.2).pow(2)) as f32).sqrt()
    }
}
