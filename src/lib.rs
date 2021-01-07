use std::cmp::{max, min, Ordering, Reverse};
use std::hash::Hash;
use std::ops::Index;

use ahash::AHashSet as HashSet;
#[cfg(feature = "indicatif")]
use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Hnsw<P> {
    ef_search: usize,
    points: Vec<P>,
    zero: Vec<ZeroNode>,
    layers: Vec<Vec<UpperNode>>,
}

impl<P> Hnsw<P>
where
    P: Point,
{
    pub fn builder() -> Builder {
        Builder::default()
    }

    fn new(points: &[P], builder: Builder) -> (Self, Vec<PointId>) {
        let ef_search = builder.ef_search.unwrap_or(100);
        let ef_construction = builder.ef_construction.unwrap_or(100);
        let ml = builder.ml.unwrap_or_else(|| (M as f32).ln());
        #[cfg(feature = "indicatif")]
        let progress = builder.progress;
        #[cfg(feature = "indicatif")]
        if let Some(bar) = &progress {
            bar.set_draw_delta(1_000);
            bar.set_length(points.len() as u64);
        }

        if points.is_empty() {
            return (
                Self {
                    ef_search,
                    zero: Vec::new(),
                    points: Vec::new(),
                    layers: Vec::new(),
                },
                Vec::new(),
            );
        }

        // Give all points a random layer and sort the list of nodes by descending order for
        // construction. This allows us to copy higher layers to lower layers as construction
        // progresses, while preserving randomness in each point's layer and insertion order.

        assert!(points.len() < u32::MAX as usize);
        let mut rng = SmallRng::from_entropy();
        let mut nodes = (0..points.len())
            .map(|i| (LayerId::random(ml, &mut rng), i))
            .collect::<Vec<_>>();
        nodes.sort_unstable_by_key(|&n| Reverse(n));

        // Sort the original `points` in layer order.
        // TODO: maybe optimize this? https://crates.io/crates/permutation

        let mut new_points = Vec::with_capacity(points.len());
        let mut new_nodes = Vec::with_capacity(points.len());
        let mut out = vec![PointId::invalid(); points.len()];
        for (i, &(layer, idx)) in nodes.iter().enumerate() {
            let pid = PointId(i as u32);
            new_points.push(points[idx].clone());
            new_nodes.push((layer, pid));
            out[idx] = pid;
        }
        let (points, nodes) = (new_points, new_nodes);

        // The layer from the first node is our top layer, or the zero layer if we have no nodes.

        let top = match nodes.first() {
            Some((top, _)) => *top,
            None => LayerId(0),
        };

        // Figure out how many nodes will go on each layer. This helps us allocate memory capacity
        // for each layer in advance, and also helps enable batch insertion of points.

        let mut sizes = vec![0; top.0 + 1];
        for (layer, _) in nodes.iter().copied() {
            sizes[layer.0] += 1;
        }

        let mut start = 0;
        let mut ranges = Vec::with_capacity(top.0);
        for (i, size) in sizes.into_iter().enumerate().rev() {
            // Skip the first point, since we insert the enter point separately
            ranges.push((LayerId(i), max(start, 1)..start + size));
            start += size;
        }

        // Insert the first point so that we have an enter point to start searches with.

        let mut layers = vec![vec![]; top.0];
        let mut zero = Vec::with_capacity(points.len());
        zero.push(ZeroNode::default());

        let mut pool = SearchPool::default();
        let mut batch = Vec::new();
        let mut done = Vec::new();
        let max_batch_len = num_cpus::get() * 4;
        for (layer, mut range) in ranges {
            let num = if layer.0 > 0 { M } else { M * 2 };

            while range.start < range.end {
                let len = min(range.len(), max_batch_len);
                batch.clear();
                batch.extend(
                    nodes[range.start..(range.start + len)]
                        .iter()
                        .map(|&(_, pid)| (pid, pool.pop())),
                );

                batch.par_iter_mut().for_each(|(pid, search)| {
                    let point = &points[*pid];
                    search.push(PointId(0), point, &points);
                    for cur in top.descend() {
                        search.ef = if cur <= layer { ef_construction } else { 1 };
                        zero.search(point, search, &points, num);
                        match cur > layer {
                            true => search.cull(),
                            false => break,
                        }
                    }
                });

                done.clear();
                for (pid, mut search) in batch.drain(..) {
                    for added in done.iter().copied() {
                        search.push(added, &points[pid], &points);
                    }
                    insert(&mut zero, pid, &search.nearest, &points);
                    done.push(pid);
                    pool.push(search);
                }

                #[cfg(feature = "indicatif")]
                if let Some(bar) = &progress {
                    bar.inc(done.len() as u64);
                }

                range.start += len;
            }

            // For layers above the zero layer, make a copy of the current state of the zero layer
            // with `nearest` truncated to `M` elements.
            if layer.0 > 0 {
                let mut upper = Vec::with_capacity(zero.len());
                upper.extend(zero.iter().map(|zero| {
                    let mut upper = UpperNode::default();
                    upper.nearest.copy_from_slice(&zero.nearest[..M]);
                    upper
                }));
                layers[layer.0 - 1] = upper;
            }
        }

        #[cfg(feature = "indicatif")]
        if let Some(bar) = progress {
            bar.finish();
        }

        (
            Self {
                ef_search,
                zero,
                points,
                layers,
            },
            out,
        )
    }

    /// Search the index for the points nearest to the reference point `point`
    ///
    /// The results are returned in the `out` parameter; the number of neighbors to search for
    /// is limited by the size of the `out` parameter, and the number of results found is returned
    /// in the return value.
    pub fn search(&self, point: &P, out: &mut [PointId], search: &mut Search) -> usize {
        if self.points.is_empty() {
            return 0;
        }

        search.reset();
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            match cur.0 {
                0 => self.zero.search(point, search, &self.points, num),
                l => self.layers[l - 1].search(point, search, &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        let found = min(search.nearest.len(), out.len());
        for (i, candidate) in search.nearest[..found].iter().enumerate() {
            out[i] = candidate.pid;
        }
        found
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (PointId(i as u32), p))
    }
}

/// Insert new node in the zero layer
///
/// `new` contains the `PointId` for the new node; `found` is a slice containing all
/// `Candidate`s found during searching (ordered from near to far).
///
/// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
/// for the new node's neighbors if necessary before appending the new node to the layer.
fn insert<P: Point>(layer: &mut Vec<ZeroNode>, new: PointId, found: &[Candidate], points: &[P]) {
    let mut node = ZeroNode::default();

    // Just make sure the candidates are all unique
    debug_assert_eq!(
        found.len(),
        found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
    );

    for (i, candidate) in found.iter().take(M * 2).enumerate() {
        // `candidate` here is the new node's neighbor
        let &Candidate { distance, pid } = candidate;
        node.nearest[i] = pid; // Update the new node's `nearest`

        let old = &points[pid];
        let nearest = &layer[pid.0 as usize].nearest;

        // Find the correct index to insert at to keep the neighbor's neighbors sorted
        let idx = nearest
            .binary_search_by(|third| {
                // `third` here is one of the neighbors of the new node's neighbor.
                let third = match third {
                    pid if pid.is_valid() => *pid,
                    // if `third` is `None`, our new `node` is always "closer"
                    _ => return Ordering::Greater,
                };

                distance.cmp(&old.distance(&points[third.0 as usize]).into())
            })
            .unwrap_or_else(|e| e);

        // It might be possible for all the neighbor's current neighbors to be closer to our
        // neighbor than to the new node, in which case we skip insertion of our new node's ID.
        if idx >= nearest.len() {
            continue;
        }

        let nearest = &mut layer[pid.0 as usize].nearest;
        if !nearest[idx].is_valid() {
            nearest[idx] = new;
            continue;
        }

        let end = (M * 2) - 1;
        nearest.copy_within(idx..end, idx + 1);
        nearest[idx] = new;
    }

    layer.push(node);
}

#[derive(Default)]
struct SearchPool {
    pool: Vec<Search>,
}

impl SearchPool {
    fn pop(&mut self) -> Search {
        match self.pool.pop() {
            Some(mut search) => {
                search.reset();
                search
            }
            None => Search::default(),
        }
    }

    fn push(&mut self, search: Search) {
        self.pool.push(search);
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: HashSet<PointId>,
    /// Candidates for further inspection (`C` in the paper)
    candidates: Vec<Candidate>,
    /// Nearest neighbors found so far (`W` in the paper)
    nearest: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    ef: usize,
    /// Current furthest node in `nearest`
    furthest: OrderedFloat<f32>,
}

impl Search {
    /// Resets the state to be ready for a new search
    fn reset(&mut self) {
        let Search {
            visited,
            candidates,
            nearest,
            ef: _,
            furthest,
        } = self;

        visited.clear();
        candidates.clear();
        nearest.clear();
        *furthest = OrderedFloat::from(f32::INFINITY);
    }

    /// Track node `pid` as a potential new neighbor for the given `point`
    ///
    /// Will immediately return if the node has been considered before. This implements
    /// the inner loop from the paper's algorithm 2.
    fn push<P: Point>(&mut self, pid: PointId, point: &P, points: &[P]) {
        if !self.visited.insert(pid) {
            return;
        }

        let other = &points[pid];
        let distance = OrderedFloat::from(point.distance(other));
        if self.nearest.len() >= self.ef && distance > self.furthest {
            return;
        }

        if self.nearest.len() > self.ef * 2 {
            self.nearest.sort_unstable();
            self.nearest.truncate(self.ef);
            self.furthest = self.nearest.last().unwrap().distance;
        }

        let new = Candidate { distance, pid };
        self.candidates.push(new);
        self.nearest.push(new);
        self.furthest = max(self.furthest, distance);
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    ///
    /// Invariant: `nearest` should be sorted before this is called. This is generally the case
    /// because `Layer::search()` is always called right before calling `cull()`.
    fn cull(&mut self) {
        self.nearest.truncate(self.ef); // Limit size of the set of nearest neighbors
        self.furthest = self.nearest.last().unwrap().distance;
        self.candidates.clear();
        self.candidates.extend(&self.nearest);
        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: HashSet::new(),
            candidates: Vec::new(),
            nearest: Vec::new(),
            ef: 1,
            furthest: OrderedFloat::from(f32::INFINITY),
        }
    }
}

/// Parameters for building the `Hnsw`
#[derive(Default)]
pub struct Builder {
    ef_search: Option<usize>,
    ef_construction: Option<usize>,
    ml: Option<f32>,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
}

impl Builder {
    /// Set the `efConstruction` parameter from the paper
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = Some(ef_construction);
        self
    }

    /// Set the `ef` parameter from the paper
    ///
    /// If the `efConstruction` parameter is not already set, it will be set
    /// to the same value as `ef` by default.
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        if self.ef_construction.is_none() {
            self.ef_construction = Some(ef);
        }
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = Some(ml);
        self
    }

    /// A `ProgressBar` to track `Hnsw` construction progress
    #[cfg(feature = "indicatif")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }

    /// Build the `Hnsw` with the given set of points
    pub fn build<P: Point>(self, points: &[P]) -> (Hnsw<P>, Vec<PointId>) {
        Hnsw::new(points, self)
    }
}

impl Layer for Vec<ZeroNode> {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].nearest,
        }
    }
}

impl Layer for Vec<UpperNode> {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].nearest,
        }
    }
}

trait Layer {
    /// Search this layer for nodes near the given `point`
    ///
    /// This contains the loops from the paper's algorithm 2. `point` represents `q`, the query
    /// element; `search.candidates` contains the enter points `ep`. `points` contains all the
    /// points, which is required to calculate distances between two points.
    ///
    /// The `links` argument represents the number of links from each candidate to consider. This
    /// function may be called for a higher layer (with M links per node) or the zero layer (with
    /// M * 2 links per node), but for performance reasons we often call this function on the data
    /// representation matching the zero layer even when we're referring to a higher layer. In that
    /// case, we use `links` to constrain the number of per-candidate links we consider for search.
    fn search<P: Point>(&self, point: &P, search: &mut Search, points: &[P], links: usize) {
        while let Some(candidate) = search.candidates.pop() {
            if let Some(found) = search.nearest.last() {
                if candidate.distance > found.distance {
                    break;
                }
            }

            for pid in self.nearest_iter(candidate.pid).take(links) {
                search.push(pid, point, points);
            }
        }

        search.nearest.sort_unstable();
        search.nearest.truncate(search.ef);
    }

    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_>;
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default)]
struct UpperNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [PointId; M],
}

impl Node for UpperNode {
    fn nearest_mut(&mut self) -> &mut [PointId] {
        &mut self.nearest
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default)]
struct ZeroNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [PointId; M * 2],
}

impl Node for ZeroNode {
    fn nearest_mut(&mut self) -> &mut [PointId] {
        &mut self.nearest
    }
}

trait Node: Default {
    fn nearest_mut(&mut self) -> &mut [PointId];
}

struct NearestIter<'a> {
    nearest: &'a [PointId],
}

impl<'a> Iterator for NearestIter<'a> {
    type Item = PointId;

    fn next(&mut self) -> Option<Self::Item> {
        let (&first, rest) = self.nearest.split_first()?;
        if !first.is_valid() {
            return None;
        }
        self.nearest = rest;
        Some(first)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LayerId(usize);

impl LayerId {
    fn random(ml: f32, rng: &mut SmallRng) -> Self {
        let layer = rng.next_u32() as f32 / u32::MAX as f32;
        LayerId((-(layer.ln() * ml)).floor() as usize)
    }

    fn descend(&self) -> DescendingLayerIter {
        DescendingLayerIter { next: Some(self.0) }
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

struct DescendingLayerIter {
    next: Option<usize>,
}

impl Iterator for DescendingLayerIter {
    type Item = LayerId;

    fn next(&mut self) -> Option<Self::Item> {
        Some(LayerId(match self.next? {
            0 => {
                self.next = None;
                0
            }
            next => {
                self.next = Some(next - 1);
                next
            }
        }))
    }
}

pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Candidate {
    distance: OrderedFloat<f32>,
    pid: PointId,
}

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(u32);

impl PointId {
    fn invalid() -> Self {
        PointId(u32::MAX)
    }

    /// Whether this value represents a valid point
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl Default for PointId {
    fn default() -> Self {
        PointId::invalid()
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for Vec<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl Index<PointId> for [ZeroNode] {
    type Output = ZeroNode;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let (hnsw, pids) = Hnsw::<Point>::builder().build(&[
            Point(0.1, 0.4),
            Point(-0.324, 0.543),
            Point(0.87, -0.33),
            Point(0.452, 0.932),
        ]);

        let mut search = Search::default();
        let mut results = vec![PointId::invalid()];
        let p = Point(0.1, 0.35);
        let found = hnsw.search(&p, &mut results, &mut search);
        assert_eq!(found, 1);
        assert_eq!(&results, &[pids[0]]);
    }

    #[derive(Clone, Copy, Debug)]
    struct Point(f32, f32);

    impl super::Point for Point {
        fn distance(&self, other: &Self) -> f32 {
            ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
        }
    }
}
