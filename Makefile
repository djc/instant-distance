instant-distance-py/test/instant_distance.so: instant-distance-py/src/lib.rs
	RUSTFLAGS="-C target-cpu=native" cargo build --release
	([ -f target/release/libinstant_distance_py.dylib ] && cp target/release/libinstant_distance_py.dylib instant-distance-py/test/instant_distance.so) || \
	([ -f target/release/libinstant_distance_py.so ] && cp target/release/libinstant_distance_py.so instant-distance-py/test/instant_distance.so)

test-python: instant-distance-py/test/instant_distance.so
	PYTHONPATH=instant-distance-py/test/ python3 -m test

clean:
	cargo clean
	rm -f instant-distance-py/test/instant_distance.so
