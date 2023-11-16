// borrow_deref_ref doesn't get macro detection right, allow for now
#![allow(clippy::from_iter_instead_of_collect, clippy::borrow_deref_ref)]

use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::iter::{repeat, FromIterator};
use std::ops::Index;

use aligned_vec::{AVec, ConstAlign};
use instant_distance::{Len, Metric, PointId};
use pyo3::conversion::IntoPy;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyModule, PyString};
use pyo3::{pyclass, pymethods, pymodule};
use pyo3::{Py, PyAny, PyErr, PyObject, PyRef, PyRefMut, PyResult, Python};
use serde::{Deserialize, Serialize};

#[pymodule]
#[pyo3(name = "instant_distance")]
fn instant_distance_py(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Neighbor>()?;
    m.add_class::<Heuristic>()?;
    m.add_class::<Config>()?;
    m.add_class::<Search>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<HnswMap>()?;
    Ok(())
}

#[pyclass]
struct HnswMap {
    inner: instant_distance::HnswMap<[f32], EuclidMetric, MapValue, PointStorage>,
}

#[pymethods]
impl HnswMap {
    /// Build the index
    #[staticmethod]
    fn build(points: &PyList, values: &PyList, config: &Config) -> PyResult<Self> {
        let points = points
            .into_iter()
            .map(|v| {
                v.iter()?
                    .into_iter()
                    .map(|x| x?.extract())
                    .collect::<Result<Vec<_>, PyErr>>()
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        let values = values
            .into_iter()
            .map(MapValue::try_from)
            .collect::<Result<Vec<_>, PyErr>>()?;

        let hsnw_map = instant_distance::Builder::from(config)
            .build::<Vec<_>, [f32], EuclidMetric, MapValue, PointStorage>(points, values);
        Ok(Self { inner: hsnw_map })
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw_map = bincode::deserialize_from::<
            _,
            instant_distance::HnswMap<[f32], EuclidMetric, MapValue, PointStorage>,
        >(BufReader::with_capacity(
            32 * 1024 * 1024,
            File::open(fname)?,
        ))
        .map_err(|e| PyValueError::new_err(format!("deserialization error: {e:?}")))?;
        Ok(Self { inner: hnsw_map })
    }

    /// Dump the index to the given file name
    fn dump(&self, fname: &str) -> PyResult<()> {
        let f = BufWriter::with_capacity(32 * 1024 * 1024, File::create(fname)?);
        bincode::serialize_into(f, &self.inner)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e:?}")))?;
        Ok(())
    }

    /// Search the index for points neighboring the given point
    ///
    /// The `search` object contains buffers used for searching. When the search completes,
    /// iterate over the `Search` to get the results. The number of results should be equal
    /// to the `ef_search` parameter set in the index's `config`.
    ///
    /// For best performance, reusing `Search` objects is recommended.
    fn search(slf: Py<Self>, point: &PyAny, search: &mut Search, py: Python<'_>) -> PyResult<()> {
        let point = try_avec_from_py(point)?;
        let _ = slf.try_borrow(py)?.inner.search(&point, &mut search.inner);
        search.cur = Some((HnswType::Map(slf.clone_ref(py)), 0));
        Ok(())
    }
}

/// An instance of hierarchical navigable small worlds
///
/// For now, this is specialized to only support 300-element (32-bit) float vectors
/// with a squared Euclidean distance metric.
#[pyclass]
struct Hnsw {
    inner: instant_distance::Hnsw<[f32], EuclidMetric, PointStorage>,
}

#[pymethods]
impl Hnsw {
    /// Build the index
    #[staticmethod]
    fn build(input: &PyList, config: &Config) -> PyResult<(Self, Vec<u32>)> {
        let points = input
            .into_iter()
            .map(|v| {
                v.iter()?
                    .into_iter()
                    .map(|x| x?.extract())
                    .collect::<Result<Vec<_>, PyErr>>()
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        let (inner, ids) = instant_distance::Builder::from(config)
            .build_hnsw::<Vec<f32>, [f32], EuclidMetric, PointStorage>(points);
        let ids = Vec::from_iter(ids.into_iter().map(|pid| pid.into_inner()));
        Ok((Self { inner }, ids))
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw = bincode::deserialize_from::<
            _,
            instant_distance::Hnsw<[f32], EuclidMetric, PointStorage>,
        >(BufReader::with_capacity(
            32 * 1024 * 1024,
            File::open(fname)?,
        ))
        .map_err(|e| PyValueError::new_err(format!("deserialization error: {e:?}")))?;
        Ok(Self { inner: hnsw })
    }

    /// Dump the index to the given file name
    fn dump(&self, fname: &str) -> PyResult<()> {
        let f = BufWriter::with_capacity(32 * 1024 * 1024, File::create(fname)?);
        bincode::serialize_into(f, &self.inner)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e:?}")))?;
        Ok(())
    }

    /// Search the index for points neighboring the given point
    ///
    /// The `search` object contains buffers used for searching. When the search completes,
    /// iterate over the `Search` to get the results. The number of results should be equal
    /// to the `ef_search` parameter set in the index's `config`.
    ///
    /// For best performance, reusing `Search` objects is recommended.
    fn search(slf: Py<Self>, point: &PyAny, search: &mut Search, py: Python<'_>) -> PyResult<()> {
        let point = try_avec_from_py(point)?;
        let _ = slf.try_borrow(py)?.inner.search(&point, &mut search.inner);
        search.cur = Some((HnswType::Hnsw(slf.clone_ref(py)), 0));
        Ok(())
    }
}

/// Search buffer and result set
#[pyclass]
struct Search {
    inner: instant_distance::Search<[f32], EuclidMetric>,
    cur: Option<(HnswType, usize)>,
}

#[pymethods]
impl Search {
    /// Initialize an empty search buffer
    #[new]
    fn new() -> Self {
        Self {
            inner: instant_distance::Search::default(),
            cur: None,
        }
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Return the next closest point
    fn __next__(mut slf: PyRefMut<Self>) -> Option<Neighbor> {
        let (index, idx) = match slf.cur.take() {
            Some(x) => x,
            None => return None,
        };

        let py = slf.py();
        let neighbor = match &index {
            HnswType::Hnsw(hnsw) => {
                let hnsw = hnsw.as_ref(py).borrow();
                let item = hnsw.inner.get(idx, &slf.inner);
                item.map(|item| Neighbor {
                    distance: item.distance,
                    pid: item.pid.into_inner(),
                    value: py.None(),
                })
            }
            HnswType::Map(map) => {
                let map = map.as_ref(py).borrow();
                let item = map.inner.get(idx, &slf.inner);
                item.map(|item| Neighbor {
                    distance: item.distance,
                    pid: item.pid.into_inner(),
                    value: item.value.into_py(py),
                })
            }
        };

        slf.cur = neighbor.as_ref().map(|_| (index, idx + 1));
        neighbor
    }
}

enum HnswType {
    Hnsw(Py<Hnsw>),
    Map(Py<HnswMap>),
}

#[pyclass]
#[derive(Copy, Clone, Default)]
struct Config {
    /// Number of nearest neighbors to cache during the search
    #[pyo3(get, set)]
    ef_search: usize,
    /// Number of nearest neighbors to cache during construction
    #[pyo3(get, set)]
    ef_construction: usize,
    /// Parameter to control the number of layers
    #[pyo3(get, set)]
    ml: f32,
    /// Random seed used to randomize the order of points
    ///
    /// This can be useful if you want to have fully deterministic results.
    #[pyo3(get, set)]
    seed: u64,
    /// Whether to use the heuristic search algorithm
    ///
    /// This will prioritize neighbors that are farther away from other, closer neighbors,
    /// in order to get better results on clustered data points.
    #[pyo3(get, set)]
    heuristic: Option<Heuristic>,
}

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        let builder = instant_distance::Builder::default();
        let (ef_search, ef_construction, ml, seed) = builder.into_parts();
        let heuristic = Some(Heuristic::default());
        Self {
            ef_search,
            ef_construction,
            ml,
            seed,
            heuristic,
        }
    }
}

impl From<&Config> for instant_distance::Builder {
    fn from(py: &Config) -> Self {
        let Config {
            ef_search,
            ef_construction,
            ml,
            seed,
            heuristic,
        } = *py;
        Self::default()
            .ef_search(ef_search)
            .ef_construction(ef_construction)
            .ml(ml)
            .seed(seed)
            .select_heuristic(heuristic.map(|h| h.into()))
    }
}

#[pyclass]
#[derive(Copy, Clone)]
struct Heuristic {
    /// Whether to extend the candidate set before selecting results
    ///
    /// This is only useful only for extremely clustered data.
    #[pyo3(get, set)]
    extend_candidates: bool,
    /// Whether to keep pruned neighbors to make the neighbor set size constant
    #[pyo3(get, set)]
    keep_pruned: bool,
}

#[pymethods]
impl Heuristic {
    #[new]
    fn new() -> Self {
        let default = instant_distance::Heuristic::default();
        let instant_distance::Heuristic {
            extend_candidates,
            keep_pruned,
        } = default;
        Self {
            extend_candidates,
            keep_pruned,
        }
    }
}

impl Default for Heuristic {
    fn default() -> Self {
        Self {
            extend_candidates: false,
            keep_pruned: true,
        }
    }
}

impl From<Heuristic> for instant_distance::Heuristic {
    fn from(py: Heuristic) -> Self {
        let Heuristic {
            extend_candidates,
            keep_pruned,
        } = py;
        Self {
            extend_candidates,
            keep_pruned,
        }
    }
}

/// Item found by the nearest neighbor search
#[pyclass]
struct Neighbor {
    /// Distance to the neighboring point
    #[pyo3(get)]
    distance: f32,
    /// Identifier for the neighboring point
    #[pyo3(get)]
    pid: u32,
    /// Value for the neighboring point (only set for `HnswMap` results)
    #[pyo3(get)]
    value: PyObject,
}

#[pymethods]
impl Neighbor {
    fn __repr__(&self) -> PyResult<String> {
        match Python::with_gil(|py| self.value.is_none(py)) {
            false => Ok(format!(
                "instant_distance.Neighbor(distance={}, pid={}, value={})",
                self.distance,
                self.pid,
                Python::with_gil(|py| self.value.as_ref(py).repr().map(|s| s.to_string()))?,
            )),
            true => Ok(format!(
                "instant_distance.Item(distance={}, pid={})",
                self.distance, self.pid,
            )),
        }
    }
}

fn try_avec_from_py(value: &PyAny) -> Result<AVec<f32, ConstAlign<ALIGNMENT>>, PyErr> {
    let mut new = AVec::new(ALIGNMENT);
    for val in value.iter()? {
        new.push(val?.extract::<f32>()?);
    }
    for _ in 0..PointStorage::padding(new.len()) {
        new.push(0.0);
    }
    Ok(new)
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct EuclidMetric;

impl Metric<[f32]> for EuclidMetric {
    fn distance(lhs: &[f32], rhs: &[f32]) -> f32 {
        debug_assert_eq!(lhs.len(), rhs.len());

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{
                _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_load_ps,
                _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
                _mm_movehl_ps, _mm_shuffle_ps,
            };
            debug_assert_eq!(lhs.len() % 8, 0);

            unsafe {
                let mut acc_8x = _mm256_setzero_ps();
                for (lh_slice, rh_slice) in lhs.chunks_exact(8).zip(rhs.chunks_exact(8)) {
                    let lh_8x = _mm256_load_ps(lh_slice.as_ptr());
                    let rh_8x = _mm256_load_ps(rh_slice.as_ptr());
                    let diff = _mm256_sub_ps(lh_8x, rh_8x);
                    acc_8x = _mm256_fmadd_ps(diff, diff, acc_8x);
                }

                let mut acc_4x = _mm256_extractf128_ps(acc_8x, 1); // upper half
                let right = _mm256_castps256_ps128(acc_8x); // lower half
                acc_4x = _mm_add_ps(acc_4x, right); // sum halves

                let lower = _mm_movehl_ps(acc_4x, acc_4x);
                acc_4x = _mm_add_ps(acc_4x, lower);
                let upper = _mm_shuffle_ps(acc_4x, acc_4x, 0x1);
                acc_4x = _mm_add_ss(acc_4x, upper);
                _mm_cvtss_f32(acc_4x)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        lhs.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PointStorage {
    point_len: usize,
    points_data: AVec<f32>,
}

impl PointStorage {
    const fn padding(len: usize) -> usize {
        let floats_per_alignment = ALIGNMENT / std::mem::size_of::<f32>();
        match len % floats_per_alignment {
            0 => 0,
            floats_over_alignment => floats_per_alignment - floats_over_alignment,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.points_data.chunks_exact(self.point_len)
    }
}

impl Default for PointStorage {
    fn default() -> Self {
        Self {
            point_len: 1,
            points_data: AVec::new(ALIGNMENT),
        }
    }
}

impl Index<usize> for PointStorage {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let raw_idx = index * self.point_len;
        &self.points_data[raw_idx..(raw_idx + self.point_len)]
    }
}

impl Index<PointId> for PointStorage {
    type Output = [f32];

    fn index(&self, index: PointId) -> &Self::Output {
        self.index(index.into_inner() as usize)
    }
}

impl From<Vec<Vec<f32>>> for PointStorage {
    fn from(value: Vec<Vec<f32>>) -> Self {
        if let Some(point) = value.first() {
            let point_len = point.len();
            let padding = PointStorage::padding(point_len);
            let mut points_data =
                AVec::with_capacity(ALIGNMENT, value.len() * (point_len + padding));
            for point in value {
                // all points should have the same length
                debug_assert_eq!(point.len(), point_len);
                for v in point.into_iter().chain(repeat(0.0).take(padding)) {
                    points_data.push(v);
                }
            }
            Self {
                point_len: point_len + padding,
                points_data,
            }
        } else {
            Default::default()
        }
    }
}

impl Len for PointStorage {
    fn len(&self) -> usize {
        self.points_data.len() / self.point_len
    }
}

impl<'a> IntoIterator for &'a PointStorage {
    type Item = &'a [f32];

    type IntoIter = PointStorageIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PointStorageIterator {
            storage: self,
            next_idx: 0,
        }
    }
}

pub struct PointStorageIterator<'a> {
    storage: &'a PointStorage,
    next_idx: usize,
}

impl<'a> Iterator for PointStorageIterator<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx < self.storage.len() {
            let result = &self.storage[self.next_idx];
            self.next_idx += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
enum MapValue {
    String(String),
}

impl TryFrom<&PyAny> for MapValue {
    type Error = PyErr;

    fn try_from(value: &PyAny) -> Result<Self, Self::Error> {
        Ok(MapValue::String(value.extract::<String>()?))
    }
}

impl IntoPy<Py<PyAny>> for &'_ MapValue {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            MapValue::String(s) => PyString::new(py, s).into(),
        }
    }
}

const ALIGNMENT: usize = 32;
