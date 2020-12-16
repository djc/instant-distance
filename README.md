![Cover logo](./cover.svg)

# Instant Distance: fast HNSW indexing

[![Build status](https://github.com/InstantDomainSearch/instant-distance/workflows/CI/badge.svg)](https://github.com/djc/quinn/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

Instance Distance is a fast pure-Rust implementation of the
[Hierarchical Navigable Small Worlds paper][paper] by Malkov and Yashunin
for finding approximate nearest neighbors. This implementation will power the
[InstantDomainSearch.com][ids] backend services used for word vector indexing.

[paper]: https://arxiv.org/abs/1603.09320
[ids]: https://instantdomainsearch.com/
