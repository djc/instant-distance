import instant_distance
import aiohttp
import asyncio
from progress.bar import IncrementalBar
from progress.spinner import Spinner


MAX_LINES = 100_000


vector_paths = {
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec": "./data/wiki.en.align.trimmed.vec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec": "./data/wiki.fr.align.trimmed.vec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.it.align.vec": "./data/wiki.it.align.trimmed.vec",
}


async def main():
    id_config = instant_distance.Config()

    print("Downloading vector files and building indexes...")
    async with aiohttp.ClientSession() as session:
        for lineno, (url, path) in enumerate(vector_paths.items()):
            points = []
            values = []
            with IncrementalBar(f"Downloading {path}", max=MAX_LINES) as bar:
                async with session.get(url) as resp:
                    with open(path, "w") as fd:
                        while True:
                            lineno += 1
                            bar.next()

                            line = await resp.content.readline()
                            if not line:
                                # EOF
                                break

                            # We just use the top 100k embeddings to
                            # save on space and time
                            if lineno >= MAX_LINES:
                                break

                            if lineno == 1:
                                # 100,000 embeddings, 300 dimensions
                                fd.write("100000 300\n")
                            else:
                                linestr = line.decode("utf-8")
                                tokens = linestr.split(" ")

                                # We track values here to build the instant distance index
                                values.append(tokens[0])
                                vec = tokens[1:]
                                points.append([float(p) for p in vec])

                                # Write out the data to our original .vec files
                                fd.write(linestr)

            # Build the instant-distance index and dump it out to a file with .idx suffix
            print("Building index...")
            hnsw = instant_distance.HnswMap.build(points, values, id_config)
            hnsw.dump(path.replace(".vec", ".idx"))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
