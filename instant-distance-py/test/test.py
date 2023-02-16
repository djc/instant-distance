import instant_distance, random


def test_hsnw(distance_metric=instant_distance.DistanceMetric.Euclid):
    points = [[random.random() for _ in range(300)] for _ in range(1024)]
    config = instant_distance.Config()
    config.distance_metric = distance_metric
    (hnsw, ids) = instant_distance.Hnsw.build(points, config)
    p = [random.random() for _ in range(300)]
    search = instant_distance.Search()
    hnsw.search(p, search)
    for candidate in search:
        print(candidate)


def test_hsnw_map(distance_metric=instant_distance.DistanceMetric.Euclid):
    the_chosen_one = 123

    embeddings = [[random.random() for _ in range(300)] for _ in range(1024)]
    with open("/usr/share/dict/words", "r") as f:  # *nix only
        values = f.read().splitlines()[1024:]

    config = instant_distance.Config()
    config.distance_metric = distance_metric
    hnsw_map = instant_distance.HnswMap.build(embeddings, values, config)

    search = instant_distance.Search()
    hnsw_map.search(embeddings[the_chosen_one], search)
    first = next(search)

    approx_nearest = first.value
    actual_word = values[the_chosen_one]

    print("approx word:\t", approx_nearest)
    print("actual word:\t", actual_word)

    assert approx_nearest == actual_word


if __name__ == "__main__":
    test_hsnw()
    test_hsnw_map()
    test_hsnw(instant_distance.DistanceMetric.Cosine)
    test_hsnw_map(instant_distance.DistanceMetric.Cosine)
