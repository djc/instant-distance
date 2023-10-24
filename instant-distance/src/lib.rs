use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::ops::Range;
#[cfg(feature = "indicatif")]
use std::sync::atomic::{self, AtomicUsize};

#[cfg(feature = "indicatif")]
use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod contiguous;
use contiguous::ContiguousStorage;
pub mod simd;
pub use contiguous::{PointRef, Storage};
mod types;
use simd::{distance_simd_f32, distance_simd_f64};
pub use types::PointId;
use types::{Candidate, Layer, LayerId, LayerSliceMut, Meta, Visited, ZeroNode, INVALID};

#[derive(Clone)]
/// Parameters for building the `Hnsw`
pub struct Builder {
    ef_search: usize,
    ef_construction: usize,
    heuristic: Option<Heuristic>,
    ml: f32,
    seed: u64,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
}

impl Builder {
    /// Set the `efConstruction` parameter from the paper
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the `ef` parameter from the paper
    ///
    /// If the `efConstruction` parameter is not already set, it will be set
    /// to the same value as `ef` by default.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn select_heuristic(mut self, params: Option<Heuristic>) -> Self {
        self.heuristic = params;
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `1.0 / ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = ml;
        self
    }

    /// Set the seed value for the random number generator used to generate a layer for each point
    ///
    /// If this value is left unset, a seed is generated from entropy (via `getrandom()`).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// A `ProgressBar` to track `Hnsw` construction progress
    #[cfg(feature = "indicatif")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }

    /// Build an `HnswMap` with the given sets of points and values
    pub fn build<E: Element, P: Point<Element = E>, V: Clone>(
        self,
        points: Vec<P>,
        values: Vec<V>,
    ) -> HnswMap<E, P, V> {
        HnswMap::new(points, values, self)
    }

    /// Build the `Hnsw` with the given set of points
    pub fn build_hnsw<E: Element, P: Point<Element = E>>(
        self,
        points: Vec<P>,
    ) -> (Hnsw<E, P>, Vec<PointId>) {
        Hnsw::new(points, self)
    }

    #[doc(hidden)]
    pub fn into_parts(self) -> (usize, usize, f32, u64) {
        let Self {
            ef_search,
            ef_construction,
            heuristic: _,
            ml,
            seed,
            ..
        } = self;
        (ef_search, ef_construction, ml, seed)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            ef_search: 100,
            ef_construction: 100,
            heuristic: Some(Heuristic::default()),
            ml: 1.0 / (M as f32).ln(),
            seed: rand::random(),
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
pub struct HnswMap<E: Element, P: Point<Element = E>, V> {
    pub hnsw: Hnsw<E, P>,
    pub values: Vec<V>,
}

impl<'a, E: Element + 'a, P: Point<Element = E> + 'a, V: Clone> HnswMap<E, P, V> {
    fn new(points: Vec<P>, values: Vec<V>, builder: Builder) -> Self {
        let (hnsw, ids) = Hnsw::new(points, builder);

        let mut sorted = ids.into_iter().enumerate().collect::<Vec<_>>();
        sorted.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        let new = sorted
            .into_iter()
            .map(|(src, _)| values[src].clone())
            .collect();

        Self { hnsw, values: new }
    }

    pub fn search<'b: 'a>(
        &'a self,
        point: &'b P,
        search: &'a mut Search,
    ) -> impl Iterator<Item = MapItem<'a, E, P, V>> + ExactSizeIterator + 'a {
        let r = point.as_slice();
        let point_ref = PointRef::from_data(r);
        self.search_by_ref(&point_ref, search)
    }

    pub fn search_by_ref(
        &'a self,
        point: &PointRef<'a, E, P>,
        search: &'a mut Search,
    ) -> impl Iterator<Item = MapItem<'a, E, P, V>> + ExactSizeIterator + 'a {
        self.hnsw
            .search(point, search)
            .map(move |item| MapItem::from(item, self))
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&'a self) -> impl Iterator<Item = (PointId, PointRef<'_, E, P>)> {
        self.hnsw.iter()
    }

    #[doc(hidden)]
    pub fn get(&self, i: usize, search: &Search) -> Option<MapItem<'_, E, P, V>> {
        Some(MapItem::from(self.hnsw.get(i, search)?, self))
    }
}

pub struct MapItem<'a, E, P: Point<Element = E>, V> {
    pub distance: f32,
    pub pid: PointId,
    pub point: PointRef<'a, E, P>,
    pub value: &'a V,
}

impl<'a, E: Element, P: Point<Element = E>, V> MapItem<'a, E, P, V> {
    fn from(item: Item<'a, E, P>, map: &'a HnswMap<E, P, V>) -> Self {
        MapItem {
            distance: item.distance,
            pid: item.pid,
            point: item.point,
            value: &map.values[item.pid.0 as usize],
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Hnsw<E: Element, P: Point<Element = E>> {
    ef_search: usize,
    pub storage: ContiguousStorage<E, P>,
    meta: Meta,
    neighbors: Vec<PointId>,
}

impl<'a, E: Element + 'a, P: Point<Element = E> + 'a> Hnsw<E, P> {
    pub fn builder() -> Builder {
        Builder::default()
    }

    fn new(points: Vec<P>, builder: Builder) -> (Self, Vec<PointId>) {
        let ef_search = builder.ef_search;
        let ef_construction = builder.ef_construction;
        let ml = builder.ml;
        let heuristic = builder.heuristic;

        #[cfg(feature = "indicatif")]
        let progress = builder.progress;
        #[cfg(feature = "indicatif")]
        if let Some(bar) = &progress {
            bar.set_length(points.len() as u64);
            bar.set_message("Build index (preparation)");
        }

        if points.is_empty() {
            return (
                Self {
                    ef_search,
                    neighbors: Vec::new(),
                    meta: Meta::default(),
                    storage: ContiguousStorage::empty(),
                },
                Vec::new(),
            );
        }

        let points_len = points.len();
        let mut meta = Meta::new(ml, points_len);

        let (storage, out, layer_assignments) =
            ContiguousStorage::<E, P>::new(points, &meta, builder);

        // Insert the first point so that we have an enter point to start searches with.

        let mut neighbors = vec![INVALID; meta.neighbors()];
        let mut layers = meta.layers_mut(&mut neighbors);
        let (zero, upper) = layers.split_first_mut().unwrap();
        let zero = zero.zero_nodes();

        let state = Construction {
            meta: &mut meta,
            zero: zero.as_slice(),
            upper,
            pool: SearchPool::new(points_len),
            storage: &storage,
            heuristic,
            ef_construction,
            #[cfg(feature = "indicatif")]
            progress,
            #[cfg(feature = "indicatif")]
            done: AtomicUsize::new(0),
        };

        for layer in state.meta.descending() {
            #[cfg(feature = "indicatif")]
            if let Some(bar) = &state.progress {
                bar.set_message(format!("Building index (layer {})", layer.0));
            }

            let Range { start, end } = state.meta.points(layer);
            layer_assignments[start..end]
                .into_par_iter()
                .for_each(|(_, pid)| {
                    state.insert(*pid, layer);
                });

            // Copy the current state of the zero layer
            match layer.0 {
                0 => break,
                n => state.upper[n - 1].copy_from_zero(&zero[..end]),
            }
        }

        #[cfg(feature = "indicatif")]
        if let Some(bar) = state.progress {
            bar.finish();
        }

        (
            Self {
                ef_search,
                neighbors,
                meta,
                storage,
            },
            out,
        )
    }

    /// Search the index for the points nearest to the reference point `point`
    ///
    /// The results are returned in the `out` parameter; the number of neighbors to search for
    /// is limited by the size of the `out` parameter, and the number of results found is returned
    /// in the return value.
    pub fn search(
        &'a self,
        point: &PointRef<'a, E, P>,
        search: &'a mut Search,
    ) -> impl Iterator<Item = Item<'a, E, P>> + ExactSizeIterator + 'a {
        search.reset();
        let map = move |candidate| Item::new(candidate, self);
        if self.storage.is_empty() {
            return search.iter().map(map);
        }

        search.visited.reserve_capacity(self.storage.len());
        search.push(PointId(0), point, &self.storage);
        for cur in self.meta.descending() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            let layer = self.meta.layer(cur, &self.neighbors);
            search.search(point, layer, &self.storage, num);

            if !cur.is_zero() {
                search.cull();
            }
        }

        search.iter().map(map)
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, PointRef<'_, E, P>)> {
        self.storage
            .iter()
            .enumerate()
            .map(|(i, p)| (PointId(i as u32), p))
    }

    #[doc(hidden)]
    pub fn get(&self, i: usize, search: &Search) -> Option<Item<'_, E, P>> {
        Some(Item::new(search.nearest.get(i).cloned()?, self))
    }
}

pub struct Item<'a, E, P: Point<Element = E>> {
    pub distance: f32,
    pub pid: PointId,
    pub point: PointRef<'a, E, P>,
}

impl<'a, E: Element, P: Point<Element = E>> Item<'a, E, P> {
    fn new(candidate: Candidate, hnsw: &'a Hnsw<E, P>) -> Self {
        Self {
            distance: candidate.distance.into_inner(),
            pid: candidate.pid,
            point: hnsw.storage.get(candidate.pid.0 as usize).unwrap(),
        }
    }
}

struct Construction<'a, E: Element, P: Point<Element = E>> {
    meta: &'a Meta,
    zero: &'a [RwLock<ZeroNode<'a>>],
    upper: &'a mut [LayerSliceMut<'a>],
    pool: SearchPool,
    storage: &'a ContiguousStorage<E, P>,
    heuristic: Option<Heuristic>,
    ef_construction: usize,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
    #[cfg(feature = "indicatif")]
    done: AtomicUsize,
}

impl<'a, E: Element, P: Point<Element = E>> Construction<'a, E, P> {
    /// Insert new node in the zero layer
    ///
    /// * `new` is the `PointId` for the new node
    /// * `layer` contains all the nodes at the current layer
    /// * `layers` refers to the existing higher-level layers
    ///
    /// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
    /// for the new node's neighbors if necessary before appending the new node to the layer.
    fn insert(&self, new: PointId, layer: LayerId) {
        let mut node = self.zero[new].write();
        let (mut search, mut insertion) = self.pool.pop();
        insertion.ef = self.ef_construction;

        let point_ref = &self.storage.get(new.0 as usize).unwrap();
        search.reset();
        search.push(PointId(0), point_ref, self.storage);
        let num = if layer.is_zero() { M * 2 } else { M };

        for cur in self.meta.descending() {
            search.ef = if cur <= layer {
                self.ef_construction
            } else {
                1
            };
            match cur > layer {
                true => {
                    search.search(point_ref, self.upper[cur.0 - 1].as_ref(), self.storage, num);
                    search.cull();
                }
                false => {
                    search.search(point_ref, self.zero, self.storage, num);
                    break;
                }
            }
        }

        let found = match self.heuristic {
            None => {
                let candidates = search.select_simple();
                &candidates[..Ord::min(candidates.len(), M * 2)]
            }
            Some(heuristic) => search.select_heuristic(
                &self.storage.get(new.0 as usize).unwrap(),
                self.zero,
                self.storage,
                heuristic,
            ),
        };

        // Just make sure the candidates are all unique
        debug_assert_eq!(
            found.len(),
            found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
        );

        for (i, candidate) in found.iter().enumerate() {
            // `candidate` here is the new node's neighbor
            let &Candidate { distance, pid } = candidate;
            if let Some(heuristic) = self.heuristic {
                let mut candidate_write_lock = self.zero[pid].write();
                let current = candidate_write_lock
                    .iter()
                    .take_while(|p| p.is_valid())
                    .copied();
                let found = insertion.add_neighbor_heuristic(
                    new,
                    current,
                    self.zero,
                    &self.storage.get(pid.0 as usize).unwrap(),
                    self.storage,
                    heuristic,
                );

                candidate_write_lock.rewrite(found.iter().map(|candidate| candidate.pid))
            } else {
                // Find the correct index to insert at to keep the neighbor's neighbors sorted
                let old = &self.storage.get(pid.0 as usize).unwrap();
                let idx = self.zero[pid]
                    .read()
                    .binary_search_by(|third| {
                        // `third` here is one of the neighbors of the new node's neighbor.
                        let third = match third {
                            pid if pid.is_valid() => *pid,
                            // if `third` is `None`, our new `node` is always "closer"
                            _ => return Ordering::Greater,
                        };

                        distance.cmp(&OrderedFloat::from(
                            old.distance(&self.storage.get(third.0 as usize).unwrap()),
                        ))
                    })
                    .unwrap_or_else(|e| e);

                self.zero[pid].write().insert(idx, new);
            }

            node.set(i, pid);
        }

        #[cfg(feature = "indicatif")]
        if let Some(bar) = &self.progress {
            let value = self.done.fetch_add(1, atomic::Ordering::Relaxed);
            if value % 1000 == 0 {
                bar.set_position(value as u64);
            }
        }

        self.pool.push((search, insertion));
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
    fn search<E: Element, L: Layer, P: Point<Element = E>>(
        &mut self,
        point_ref: &PointRef<'_, E, P>,
        layer: L,
        points: &ContiguousStorage<E, P>,
        links: usize,
    ) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for pid in layer.nearest_iter(candidate.pid).take(links) {
                self.push(pid, point_ref, points);
            }

            // If we don't truncate here, `furthest` will be further out than necessary, making
            // us continue looping while we could have broken out.
            self.nearest.truncate(self.ef);
        }
    }

    fn add_neighbor_heuristic<E: Element, L: Layer, P: Point<Element = E>>(
        &mut self,
        new: PointId,
        current: impl Iterator<Item = PointId>,
        layer: L,
        point_ref: &PointRef<'_, E, P>,
        storage: &ContiguousStorage<E, P>,
        params: Heuristic,
    ) -> &[Candidate] {
        self.reset();
        self.push(new, point_ref, storage);
        for pid in current {
            self.push(pid, point_ref, storage);
        }
        self.select_heuristic(point_ref, layer, storage, params)
    }

    /// Heuristically sort and truncate neighbors in `self.nearest`
    ///
    /// Invariant: `self.nearest` must be in sorted (nearest first) order.
    fn select_heuristic<E: Element, L: Layer, P: Point<Element = E>>(
        &mut self,
        point: &PointRef<'_, E, P>,
        layer: L,
        storage: &ContiguousStorage<E, P>,
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

                    let other = storage.get(hop.0 as usize).unwrap();
                    let distance = OrderedFloat::from(point.distance(&other));
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
            let candidate_point = storage.get(candidate.pid.0 as usize).unwrap();
            let nearest = !self.nearest.iter().any(|result| {
                let result = storage.get(result.pid.0 as usize).unwrap();
                let distance = OrderedFloat::from(candidate_point.distance(&result));
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
    fn push<E: Element, P: Point<Element = E>>(
        &mut self,
        pid: PointId,
        point_ref: &PointRef<'_, E, P>,
        storage: &ContiguousStorage<E, P>,
    ) {
        if !self.visited.insert(pid) {
            return;
        }
        let other = storage.get(pid.0 as usize).unwrap();
        let distance = OrderedFloat::from(point_ref.distance(&other));
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
    fn select_simple(&mut self) -> &[Candidate] {
        &self.nearest
    }

    fn iter(&self) -> impl Iterator<Item = Candidate> + ExactSizeIterator + '_ {
        self.nearest.iter().copied()
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

pub trait Point: Sync {
    const STRIDE: usize;
    type Element: Element;
    fn distance(&self, other: &Self) -> f32;
    fn as_slice(&self) -> &[Self::Element];
}

/// This trait provides for indirection against the underlying data type stored. This gives us some flexibility in defining implementations for different data types.
/// For example, we can implement this for f32, and f64, and then define a specific, optimized SIMD implementation for each.
pub trait Element: Sync + Copy {
    fn distance(lhs: &[Self], rhs: &[Self]) -> f32;
}

impl Element for i8 {
    fn distance(lhs: &[Self], rhs: &[Self]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (*a - *b).pow(2) as f32)
            .sum::<f32>()
            .sqrt()
    }
}

macro_rules! naive_impl {
    ($($t:ty),*) => {
        $(
            impl Element for $t {
                fn distance(lhs: &[Self], rhs: &[Self]) -> f32 {
                    lhs.iter()
                        .zip(rhs.iter())
                        .map(|(a, b)| (*a - *b).pow(2) as f32)
                        .sum::<f32>()
                        .sqrt()
                }
            }
        )*
    };
}

naive_impl!(u8, u16, u32, u64, u128, usize, i16, i32, i64, i128, isize);

impl Element for f64 {
    fn distance(lhs: &[Self], rhs: &[Self]) -> f32 {
        distance_simd_f64(lhs, rhs) as f32
    }
}

impl Element for f32 {
    fn distance(lhs: &[Self], rhs: &[Self]) -> f32 {
        distance_simd_f32(lhs, rhs)
    }
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 32;
