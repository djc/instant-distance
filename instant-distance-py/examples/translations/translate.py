import sys
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


async def download_build_index():
    id_config = instant_distance.Config()

    print("Downloading vector files and building indexes...")
    async with aiohttp.ClientSession() as session:
        for url, path in vector_paths.items():
            lineno = 0
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

                            # 100,000 embeddings, 300 dimensions
                            if lineno == 1:
                                fd.write("100000 300\n")
                            # We just use the top 100k embeddings to
                            # save on space and time
                            elif lineno > MAX_LINES:
                                break
                            else:
                                linestr = line.decode("utf-8")
                                tokens = linestr.split(" ")

                                # We track values here to build the instant-distance index
                                values.append(tokens[0])
                                vec = tokens[1:]
                                points.append([float(p) for p in vec])

                                # Write out the data to our original .vec files
                                fd.write(linestr)

            # Build the instant-distance index and dump it out to a file with .idx suffix
            print("Building index... (This will take a minute)")
            hnsw = instant_distance.HnswMap.build(points, values, id_config)
            hnsw.dump(path.replace(".vec", ".idx"))

async def translate(word):
    pass


async def main():
    args = sys.argv[1:]
    try:
        if args[0] == "prepare":
            await download_build_index()
        elif args[0] == "translate":        
            await translate(args[1])
    except IndexError:
        pass
    
    print(f"usage:\t{sys.argv[0]} prepare\n\t{sys.argv[0]} translate <english word>")
    exit(1)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

