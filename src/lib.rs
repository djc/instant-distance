use std::cmp::{max, Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::HashSet;
#[cfg(feature = "indicatif")]
use std::sync::atomic::{self, AtomicUsize};

#[cfg(feature = "indicatif")]
use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod types;
pub use types::PointId;
use types::{Candidate, Layer, LayerId, UpperNode, Visited, ZeroNode, INVALID};

/// Parameters for building the `Hnsw`
pub struct Builder {
    ef_search: Option<usize>,
    ef_construction: Option<usize>,
    heuristic: Option<Heuristic>,
    ml: Option<f32>,
    seed: Option<u64>,
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
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    pub fn select_heuristic(mut self, params: Option<Heuristic>) -> Self {
        self.heuristic = params;
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = Some(ml);
        self
    }

    /// Set the seed value for the random number generator used to generate a layer for each point
    ///
    /// If this value is left unset, a seed is generated from entropy (via `getrandom()`).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

impl Default for Builder {
    fn default() -> Self {
        Self {
            ef_search: None,
            ef_construction: None,
            heuristic: Some(Heuristic::default()),
            ml: None,
            seed: None,
            #[cfg(feature = "indicatif")]
            progress: None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Heuristic {
    pub extend_candidates: bool,
    pub keep_pruned: bool,
}

impl Default for Heuristic {
    fn default() -> Self {
        Heuristic {
            extend_candidates: false,
            keep_pruned: true,
        }
    }
}

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
        let heuristic = builder.heuristic;
        let mut rng = match builder.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        #[cfg(feature = "indicatif")]
        let progress = builder.progress;
        #[cfg(feature = "indicatif")]
        if let Some(bar) = &progress {
            bar.set_draw_delta(1_000);
            bar.set_length(points.len() as u64);
            bar.set_message("Build index (preparation)");
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
        let mut nodes = (0..points.len())
            .map(|i| (LayerId::random(ml, &mut rng), i))
            .collect::<Vec<_>>();
        nodes.sort_unstable_by_key(|&n| Reverse(n));

        // Find out how many layers are needed, so that we can discard empty layers in the next
        // step. Since layer IDs are randomly generated, there might be big gaps.

        let (mut num_layers, mut prev) = (1, nodes[0].0);
        for (layer, _) in nodes.iter() {
            if *layer != prev {
                num_layers += 1;
                prev = *layer;
            }
        }

        // Sort the original `points` in layer order.
        // TODO: maybe optimize this? https://crates.io/crates/permutation

        let mut cur_layer = LayerId(num_layers - 1);
        let mut prev_layer = nodes[0].0;
        let mut new_points = Vec::with_capacity(points.len());
        let mut new_nodes = Vec::with_capacity(points.len());
        let mut out = vec![INVALID; points.len()];
        for (i, &(layer, idx)) in nodes.iter().enumerate() {
            if prev_layer != layer {
                cur_layer = LayerId(cur_layer.0 - 1);
                prev_layer = layer;
            }

            let pid = PointId(i as u32);
            new_points.push(points[idx].clone());
            new_nodes.push((cur_layer, pid));
            out[idx] = pid;
        }
        let (points, nodes) = (new_points, new_nodes);
        debug_assert_eq!(nodes.last().unwrap().0, LayerId(0));
        debug_assert_eq!(nodes.first().unwrap().0, LayerId(num_layers - 1));

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
        let zero = points
            .iter()
            .map(|_| RwLock::new(ZeroNode::default()))
            .collect::<Vec<_>>();

        let pool = SearchPool::new(points.len());
        #[cfg(feature = "indicatif")]
        let done = AtomicUsize::new(0);
        for (layer, range) in ranges {
            let num = if layer.0 > 0 { M } else { M * 2 };
            #[cfg(feature = "indicatif")]
            if let Some(bar) = &progress {
                bar.set_message(&format!("Building index (layer {})", layer.0));
            }

            nodes[range].into_par_iter().for_each(|(_, pid)| {
                let (mut search, mut insertion) = pool.pop();
                let point = &points.as_slice()[*pid];
                search.reset();
                search.push(PointId(0), point, &points);

                for cur in top.descend() {
                    search.ef = if cur <= layer { ef_construction } else { 1 };
                    match cur > layer {
                        true => {
                            search.search(point, layers[cur.0 - 1].as_slice(), &points, num);
                            search.cull();
                        }
                        false => {
                            search.search(point, zero.as_slice(), &points, num);
                            break;
                        }
                    }
                }

                insertion.ef = ef_construction;
                insert(
                    *pid,
                    &mut insertion,
                    &mut search,
                    &zero,
                    &points,
                    &heuristic,
                );

                #[cfg(feature = "indicatif")]
                if let Some(bar) = &progress {
                    let value = done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        bar.set_position(value as u64);
                    }
                }

                pool.push((search, insertion));
            });

            // For layers above the zero layer, make a copy of the current state of the zero layer
            // with `nearest` truncated to `M` elements.
            if layer.0 > 0 {
                let mut upper = Vec::new();
                (&zero).into_par_iter()
                    .map(|zero| UpperNode::from_zero(&zero.read()))
                    .collect_into_vec(&mut upper);
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
                zero: zero.into_iter().map(|node| node.into_inner()).collect(),
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

        search.visited.reserve_capacity(self.points.len());
        search.reset();
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            match cur.0 {
                0 => search.search(point, self.zero.as_slice(), &self.points, num),
                l => search.search(point, self.layers[l - 1].as_slice(), &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        let nearest = search.select_simple(out.len());
        for (i, candidate) in nearest.iter().enumerate() {
            out[i] = candidate.pid;
        }
        nearest.len()
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
/// * `new`: the `PointId` for the new node
/// * `insertion`: a `Search` for shrinking a neighbor set (only used with heuristic neighbor selection)
/// * `search`: the result for searching potential neighbors for the new node
/// *  `layer` contains all the nodes at the current layer
/// * `points` is a slice of all the points in the index
///
/// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
/// for the new node's neighbors if necessary before appending the new node to the layer.
fn insert<P: Point>(
    new: PointId,
    insertion: &mut Search,
    search: &mut Search,
    layer: &[RwLock<ZeroNode>],
    points: &[P],
    heuristic: &Option<Heuristic>,
) {
    let mut node = layer[new].write();
    let found = match heuristic {
        None => search.select_simple(M * 2),
        Some(heuristic) => search.select_heuristic(&points[new], layer, points, *heuristic),
    };

    // Just make sure the candidates are all unique
    debug_assert_eq!(
        found.len(),
        found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
    );

    for (i, candidate) in found.iter().enumerate() {
        // `candidate` here is the new node's neighbor
        let &Candidate { distance, pid } = candidate;
        if let Some(heuristic) = heuristic {
            let found = insertion.add_neighbor_heuristic(
                new,
                layer.nearest_iter(pid),
                layer,
                &points[pid],
                points,
                *heuristic,
            );

            layer[pid]
                .write()
                .rewrite(found.iter().map(|candidate| candidate.pid));
            node.set(i, pid);
        } else {
            // Find the correct index to insert at to keep the neighbor's neighbors sorted
            let old = &points[pid];
            let idx = layer[pid]
                .read()
                .binary_search_by(|third| {
                    // `third` here is one of the neighbors of the new node's neighbor.
                    let third = match third {
                        pid if pid.is_valid() => *pid,
                        // if `third` is `None`, our new `node` is always "closer"
                        _ => return Ordering::Greater,
                    };

                    distance.cmp(&old.distance(&points[third]).into())
                })
                .unwrap_or_else(|e| e);

            layer[pid].write().insert(idx, new);
            node.set(i, pid);
        }
    }
}

struct SearchPool {
    pool: Mutex<Vec<(Search, Search)>>,
    len: usize,
}

impl SearchPool {
    fn new(len: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            len,
        }
    }

    fn pop(&self) -> (Search, Search) {
        match self.pool.lock().pop() {
            Some(res) => res,
            None => (Search::new(self.len), Search::new(self.len)),
        }
    }

    fn push(&self, item: (Search, Search)) {
        self.pool.lock().push(item);
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: Visited,
    /// Candidates for further inspection (`C` in the paper)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Nearest neighbors found so far (`W` in the paper)
    ///
    /// This must always be in sorted (nearest first) order.
    nearest: Vec<Candidate>,
    /// Working set for heuristic selection
    working: Vec<Candidate>,
    discarded: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    ef: usize,
}

impl Search {
    fn new(capacity: usize) -> Self {
        Self {
            visited: Visited::with_capacity(capacity),
            ..Default::default()
        }
    }

    /// Search the given layer for nodes near the given `point`
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
    ///
    /// Invariants: `self.nearest` should be in sorted (nearest first) order, and should be
    /// truncated to `self.ef`.
    fn search<L: Layer, P: Point>(&mut self, point: &P, layer: L, points: &[P], links: usize) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for pid in layer.nearest_iter(candidate.pid).take(links) {
                self.push(pid, point, points);
            }

            // If we don't truncate here, `furthest` will be further out than necessary, making
            // us continue looping while we could have broken out.
            self.nearest.truncate(self.ef);
        }
    }

    fn add_neighbor_heuristic<L: Layer, P: Point>(
        &mut self,
        new: PointId,
        current: impl Iterator<Item = PointId>,
        layer: L,
        point: &P,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.reset();
        self.push(new, point, points);
        for pid in current {
            self.push(pid, point, points);
        }
        self.select_heuristic(point, layer, points, params)
    }

    /// Heuristically sort and truncate neighbors in `self.nearest`
    ///
    /// Invariant: `self.nearest` must be in sorted (nearest first) order.
    fn select_heuristic<L: Layer, P: Point>(
        &mut self,
        point: &P,
        layer: L,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.working.clear();
        // Get input candidates from `self.nearest` and store them in `self.working`.
        // `self.candidates` will represent `W` from the paper's algorithm 4 for now.
        for &candidate in &self.nearest {
            self.working.push(candidate);
            if params.extend_candidates {
                for hop in layer.nearest_iter(candidate.pid) {
                    if !self.visited.insert(hop) {
                        continue;
                    }

                    let other = &points[hop];
                    let distance = OrderedFloat::from(point.distance(other));
                    let new = Candidate { distance, pid: hop };
                    self.working.push(new);
                }
            }
        }

        if params.extend_candidates {
            self.working.sort_unstable();
        }

        self.nearest.clear();
        self.discarded.clear();
        for candidate in self.working.drain(..) {
            if self.nearest.len() >= M * 2 {
                break;
            }

            // Disadvantage candidates which are closer to an existing result point than they
            // are to the query point, to facilitate bridging between clustered points.
            let candidate_point = &points[candidate.pid];
            let nearest = !self.nearest.iter().any(|result| {
                let distance = OrderedFloat::from(candidate_point.distance(&points[result.pid]));
                distance < candidate.distance
            });

            match nearest {
                true => self.nearest.push(candidate),
                false => self.discarded.push(candidate),
            }
        }

        if params.keep_pruned {
            // Add discarded connections from `working` (`Wd`) to `self.nearest` (`R`)
            for candidate in self.discarded.drain(..) {
                if self.nearest.len() >= M * 2 {
                    break;
                }
                self.nearest.push(candidate);
            }
        }

        &self.nearest
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
        let new = Candidate { distance, pid };
        let idx = match self.nearest.binary_search(&new) {
            Err(idx) if idx < self.ef => idx,
            Err(_) => return,
            Ok(_) => unreachable!(),
        };

        self.nearest.insert(idx, new);
        self.candidates.push(Reverse(new));
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    ///
    /// Invariant: `nearest` should be sorted and truncated before this is called. This is generally
    /// the case because `Layer::search()` is always called right before calling `cull()`.
    fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in self.nearest.iter() {
            self.candidates.push(Reverse(candidate));
        }

        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }

    /// Resets the state to be ready for a new search
    fn reset(&mut self) {
        let Search {
            visited,
            candidates,
            nearest,
            working,
            discarded,
            ef: _,
        } = self;

        visited.clear();
        candidates.clear();
        nearest.clear();
        working.clear();
        discarded.clear();
    }

    /// Selection of neighbors for insertion (algorithm 3 from the paper)
    fn select_simple(&mut self, num: usize) -> &[Candidate] {
        self.nearest.truncate(num);
        &self.nearest
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: Visited::with_capacity(0),
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            working: Vec::new(),
            discarded: Vec::new(),
            ef: 1,
        }
    }
}

pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 32;
