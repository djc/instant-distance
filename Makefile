instant-distance-py/test/instant_distance.so: instant-distance-py/src/lib.rs
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	([ -f target/release/libinstant_distance.dylib ] && cp target/release/libinstant_distance.dylib instant-distance-py/test/instant_distance.so) || \
	([ -f target/release/libinstant_distance.so ] && cp target/release/libinstant_distance.so instant-distance-py/test/instant_distance.so)

test-python: instant-distance-py/test/instant_distance.so
	PYTHONPATH=instant-distance-py/test/ python3 -m test

bench-python: instant-distance-py/test/instant_distance.so
	PYTHONPATH=instant-distance-py/test/ python3 -m timeit -n 10 -s 'import random, instant_distance; points = [[random.random() for _ in range(300)] for _ in range(1024)]; config = instant_distance.Config()' 'instant_distance.Hnsw.build(points, config)'
	PYTHONPATH=instant-distance-py/test/ python3 -m timeit -n 10 -s 'import random, instant_distance; points = [[random.random() for _ in range(300)] for _ in range(1024)]; config = instant_distance.Config(); config.distance_metric = instant_distance.DistanceMetric.Cosine' 'instant_distance.Hnsw.build(points, config)'

clean:
	cargo clean
	rm -f instant-distance-py/test/instant_distance.so
