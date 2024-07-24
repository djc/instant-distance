![Cover logo](./cover.svg)

# Instant Distance: fast HNSW indexing

[![Build status](https://github.com/InstantDomain/instant-distance/workflows/CI/badge.svg)](https://github.com/InstantDomain/instant-distance/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

Instance Distance is a fast pure-Rust implementation of the [Hierarchical
Navigable Small Worlds paper][paper] by Malkov and Yashunin for finding
approximate nearest neighbors. This implementation powers the
[Instant Domain Search][domains] backend services used for word vector indexing.

## What it does

Instant Distance is an implementation of a fast approximate nearest neighbor
search algorithm. The algorithm is used to find the closest point(s) to a given
point in a set. As one example, it can be used to make [simple translations][translations].

## Using the library

### Rust

```toml
[dependencies]
instant-distance = "0.5.0"
```

## Example

```rust
use instant_distance::{Builder, Search};

fn main() {
    let points = vec![Point(255, 0, 0), Point(0, 255, 0), Point(0, 0, 255)];
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
```

## Testing

Rust:

```
cargo t -p instant-distance --all-features
```

Python:

```
make test-python
```

[paper]: https://arxiv.org/abs/1603.09320
[domains]: https://instantdomains.com/search
[translations]: https://instantdomains.com/engineering/how-to-use-fasttext-for-instant-translations
