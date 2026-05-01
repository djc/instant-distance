ifeq ($(shell uname), Darwin)
	PY_EXT := dylib
else
	PY_EXT := so
endif

test-python:
	cargo build --release
	cp target/release/libinstant_distance.$(PY_EXT) instant-distance-py/test/instant_distance.so
	PYTHONPATH=instant-distance-py/test/ python3 -m test

clean:
	cargo clean
	rm -f instant-distance-py/test/instant_distance.so
