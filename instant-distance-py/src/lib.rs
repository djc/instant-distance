// unnecessary_fallible_converions doesn't get macro detection right, allow for now
// https://github.com/rust-lang/rust-clippy/issues/12039
#![allow(clippy::unnecessary_fallible_conversions)]

use std::convert::{Infallible, TryFrom};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::iter::FromIterator;

use instant_distance::Point;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyList, PyListMethods, PyModule, PyModuleMethods, PyString};
use pyo3::{pyclass, pymethods, pymodule, Bound, FromPyObject, IntoPyObject};
use pyo3::{Py, PyAny, PyErr, PyRef, PyRefMut, PyResult, Python};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[pymodule]
#[pyo3(name = "instant_distance")]
fn instant_distance_py(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    inner: instant_distance::HnswMap<FloatArray, MapValue>,
}

#[pymethods]
impl HnswMap {
    /// Build the index
    #[staticmethod]
    fn build(
        points: &Bound<'_, PyList>,
        values: &Bound<'_, PyList>,
        config: &Config,
    ) -> PyResult<Self> {
        let points = points
            .iter()
            .map(|array| FloatArray::try_from(&array))
            .collect::<Result<Vec<_>, PyErr>>()?;

        let values = values
            .into_iter()
            .map(MapValue::try_from)
            .collect::<Result<Vec<_>, PyErr>>()?;

        let hsnw_map = instant_distance::Builder::from(config).build(points, values);
        Ok(Self { inner: hsnw_map })
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw_map =
            bincode::deserialize_from::<_, instant_distance::HnswMap<FloatArray, MapValue>>(
                BufReader::with_capacity(32 * 1024 * 1024, File::open(fname)?),
            )
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
    fn search(
        slf: Py<Self>,
        point: &Bound<'_, PyAny>,
        search: &mut Search,
        py: Python<'_>,
    ) -> PyResult<()> {
        let point = FloatArray::try_from(point)?;
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
    inner: instant_distance::Hnsw<FloatArray>,
}

#[pymethods]
impl Hnsw {
    /// Build the index
    #[staticmethod]
    fn build(input: &Bound<'_, PyList>, config: &Config) -> PyResult<(Self, Vec<u32>)> {
        let points = input
            .iter()
            .map(|array| FloatArray::try_from(&array))
            .collect::<Result<Vec<_>, PyErr>>()?;

        let (inner, ids) = instant_distance::Builder::from(config).build_hnsw(points);
        let ids = Vec::from_iter(ids.into_iter().map(|pid| pid.into_inner()));
        Ok((Self { inner }, ids))
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw = bincode::deserialize_from::<_, instant_distance::Hnsw<FloatArray>>(
            BufReader::with_capacity(32 * 1024 * 1024, File::open(fname)?),
        )
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
    fn search(
        slf: Py<Self>,
        point: &Bound<'_, PyAny>,
        search: &mut Search,
        py: Python<'_>,
    ) -> PyResult<()> {
        let point = FloatArray::try_from(point)?;
        let _ = slf.try_borrow(py)?.inner.search(&point, &mut search.inner);
        search.cur = Some((HnswType::Hnsw(slf.clone_ref(py)), 0));
        Ok(())
    }
}

/// Search buffer and result set
#[pyclass]
struct Search {
    inner: instant_distance::Search,
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
        let (index, idx) = slf.cur.take()?;
        let py = slf.py();
        let neighbor = match &index {
            HnswType::Hnsw(hnsw) => {
                let hnsw = hnsw.bind(py).borrow();
                let item = hnsw.inner.get(idx, &slf.inner);
                item.map(|item| Neighbor {
                    distance: item.distance,
                    pid: item.pid.into_inner(),
                    value: py.None(),
                })
            }
            HnswType::Map(map) => {
                let map = map.bind(py).borrow();
                let item = map.inner.get(idx, &slf.inner);
                item.map(|item| Neighbor {
                    distance: item.distance,
                    pid: item.pid.into_inner(),
                    value: item.value.into_pyobject(py).unwrap().unbind(), // Infallible conversion
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

#[pyclass(skip_from_py_object)]
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

#[pyclass(skip_from_py_object)]
#[derive(Copy, Clone, FromPyObject)]
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
    value: Py<PyAny>,
}

#[pymethods]
impl Neighbor {
    fn __repr__(&self) -> PyResult<String> {
        match Python::attach(|py| self.value.is_none(py)) {
            false => Ok(format!(
                "instant_distance.Neighbor(distance={}, pid={}, value={})",
                self.distance,
                self.pid,
                Python::attach(|py| self.value.bind(py).repr().map(|s| s.to_string()))?,
            )),
            true => Ok(format!(
                "instant_distance.Item(distance={}, pid={})",
                self.distance, self.pid,
            )),
        }
    }
}

#[repr(align(32))]
#[derive(Clone, Deserialize, Serialize)]
struct FloatArray(#[serde(with = "BigArray")] [f32; DIMENSIONS]);

impl TryFrom<&Bound<'_, PyAny>> for FloatArray {
    type Error = PyErr;

    fn try_from(value: &Bound<'_, PyAny>) -> Result<Self, Self::Error> {
        let mut new = FloatArray([0.0; DIMENSIONS]);
        for (i, val) in value.try_iter()?.enumerate() {
            match i >= DIMENSIONS {
                true => return Err(PyTypeError::new_err("point array too long")),
                false => new.0[i] = val?.extract::<f32>()?,
            }
        }
        Ok(new)
    }
}

impl Point for FloatArray {
    fn distance(&self, rhs: &Self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{
                _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_load_ps,
                _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
                _mm_fmadd_ps, _mm_load_ps, _mm_movehl_ps, _mm_shuffle_ps, _mm_sub_ps,
            };
            debug_assert_eq!(self.0.len() % 8, 4);

            unsafe {
                let mut acc_8x = _mm256_setzero_ps();
                for (lh_slice, rh_slice) in self.0.chunks_exact(8).zip(rhs.0.chunks_exact(8)) {
                    let lh_8x = _mm256_load_ps(lh_slice.as_ptr());
                    let rh_8x = _mm256_load_ps(rh_slice.as_ptr());
                    let diff = _mm256_sub_ps(lh_8x, rh_8x);
                    acc_8x = _mm256_fmadd_ps(diff, diff, acc_8x);
                }

                let mut acc_4x = _mm256_extractf128_ps(acc_8x, 1); // upper half
                let right = _mm256_castps256_ps128(acc_8x); // lower half
                acc_4x = _mm_add_ps(acc_4x, right); // sum halves

                let lh_4x = _mm_load_ps(self.0[DIMENSIONS - 4..].as_ptr());
                let rh_4x = _mm_load_ps(rhs.0[DIMENSIONS - 4..].as_ptr());
                let diff = _mm_sub_ps(lh_4x, rh_4x);
                acc_4x = _mm_fmadd_ps(diff, diff, acc_4x);

                let lower = _mm_movehl_ps(acc_4x, acc_4x);
                acc_4x = _mm_add_ps(acc_4x, lower);
                let upper = _mm_shuffle_ps(acc_4x, acc_4x, 0x1);
                acc_4x = _mm_add_ss(acc_4x, upper);
                _mm_cvtss_f32(acc_4x)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
    }
}

#[derive(Clone, Deserialize, Serialize)]
enum MapValue {
    String(String),
}

impl TryFrom<Bound<'_, PyAny>> for MapValue {
    type Error = PyErr;

    fn try_from(value: Bound<'_, PyAny>) -> Result<Self, Self::Error> {
        Ok(MapValue::String(value.extract::<String>()?))
    }
}

impl<'py> IntoPyObject<'py> for &'_ MapValue {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            MapValue::String(s) => Ok(PyString::new(py, s).into_any()),
        }
    }
}

const DIMENSIONS: usize = 300;
