use instant_distance::{Builder, Search};

fn main() {
    let points = vec![Point([255, 0, 0]), Point([0, 255, 0]), Point([0, 0, 255])];
    let values = vec!["red", "green", "blue"];

    let map = Builder::default().build(points, values);
    let mut search = Search::default();

    let closest_point = map
        .search(&Point([204, 85, 0]), &mut search)
        .next()
        .unwrap();

    println!("{:?}", closest_point.value);
}

#[derive(Clone, Copy, Debug)]
struct Point([isize; 3]);

impl instant_distance::Point for Point {
    const STRIDE: usize = 3;
    type Element = isize;
    fn as_slice(&self) -> &[Self::Element] {
        &self.0
    }

    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        (((self.0[0] - other.0[0]).pow(2)
            + (self.0[1] - other.0[1]).pow(2)
            + (self.0[2] - other.0[2]).pow(2)) as f32)
            .sqrt()
    }
}
