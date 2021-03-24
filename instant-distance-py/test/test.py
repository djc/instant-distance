import instant_distance, random

def main():
    points = [[random.random() for _ in range(300)] for _ in range(1024)]
    config = instant_distance.Config()
    (hnsw, ids) = instant_distance.Hnsw.build(points, config)
    p = [random.random() for _ in range(300)]
    search = instant_distance.Search()
    hnsw.search(p, search)
    for candidate in search:
        print(candidate)

if __name__ == '__main__':
    main()
