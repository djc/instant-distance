test-python:
	cargo build --release
	cp target/release/libinstant_distance.dylib instant-distance-py/test/instant_distance.so
	PYTHONPATH=instant-distance-py/test/ python3 -m test
