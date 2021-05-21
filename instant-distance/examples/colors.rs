use instant_distance::{Builder, Search};

fn main() {
    //
    let points = vec![Point(255, 0, 0), Point(255, 0, 0), Point(255, 0, 0)];
    let values = vec!["red", "green", "blue"];

    let map = Builder::default().build(points, values);
    let mut search = Search::default();

    let cambridge_blue = Point(163, 193, 173);

    let closest_point = map.search(&cambridge_blue, &mut search).next().unwrap();

    println!("{:?}", closest_point.value);
}

#[derive(Clone, Copy, Debug)]
struct Point(isize, isize, isize);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        (((self.0 - other.0).pow(2) + (self.1 - other.1).pow(2) + (self.2 - other.2).pow(2)) as f32)
            .sqrt()
    }
}
