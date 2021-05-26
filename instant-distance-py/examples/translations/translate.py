import json
import asyncio
import os
import sys

import aiohttp
import instant_distance
from progress.bar import IncrementalBar
from progress.spinner import Spinner

MAX_LINES = 100_000
LANGS = ("en", "fr", "it")
LANG_REPLACE = "$$lang"
DL_TEMPLATE = f"https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{LANG_REPLACE}.align.vec"
BUILT_IDX_PATH = f"./data/{'_'.join(LANGS)}.idx"
WORD_MAP_PATH = f"./data/{'_'.join(LANGS)}.json"


async def download_build_index():
    """
    This function downloads pre-trained word vectors trained on Wikipedia using fastText:
    https://fasttext.cc/docs/en/aligned-vectors.html

    The content is streamed and we take only the first 100,000 lines and drop the longtail
    of less common words. We intercept each line and use this information to also build
    an instant-distance index file.
    """
    id_config = instant_distance.Config()
    points = []
    values = []

    word_map = {}

    print("Downloading vector files and building indexes...")
    async with aiohttp.ClientSession() as session:
        for lang in LANGS:
            # Construct a url for each language and path
            url = DL_TEMPLATE.replace(LANG_REPLACE, lang)

            # Ensure the directory and files exist
            os.makedirs(os.path.dirname(BUILT_IDX_PATH), exist_ok=True)

            lineno = 0
            with IncrementalBar(
                f"Downloading {url.split('/')[-1]}", max=MAX_LINES
            ) as bar:
                async with session.get(url) as resp:
                    while True:
                        lineno += 1
                        line = await resp.content.readline()
                        if not line:
                            # EOF
                            break

                        # We just use the top 100k embeddings to
                        # save on space and time
                        elif lineno > MAX_LINES:
                            break
                        else:
                            linestr = line.decode("utf-8")
                            tokens = linestr.split(" ")

                            value = tokens[0]
                            point = tokens[1:]

                            # We track values here to build the instant-distance index
                            # Every value is prepended with 2 character language code.
                            # This allows us to determine language output later.
                            values.append(lang + value)
                            points.append([float(p) for p in point])

                            # We'll dump this out to json later
                            word_map[value] = point

                            bar.next()

    # Build the instant-distance index and dump it out to a file with .idx suffix
    print("Building index... (this will take a while)")
    hnsw = instant_distance.HnswMap.build(points, values, id_config)
    hnsw.dump(BUILT_IDX_PATH)

    # Store the mapping from string to embedding in a json file
    with open(WORD_MAP_PATH, "w") as f:
        f.write(json.dumps(word_map))


async def translate(word):
    data_exists = os.path.isfile(BUILT_IDX_PATH) and os.path.isfile(WORD_MAP_PATH)
    if not data_exists:
        print("Instant Distance index not present. Building...")
        await download_build_index()

    print("Loading indexes from filesystem...")
    with open(WORD_MAP_PATH, "w") as f:
        word_map = json.loads(f.read())

    # Get an embedding for the given word
    try:
        embedding = word_map[word]
    except KeyError:
        print(f"Word not recognized: {word}")
        exit(1)

    hnsw = instant_distance.HnswMap.load(BUILT_IDX_PATH)
    search = instant_distance.Search()
    hnsw.search(embedding, search)

    for result in list(search)[:10]:
        value = hnsw.values[result.pid]
        print(f"Language: {value[:2]}, Translation: {value[2:]}")


async def main():
    args = sys.argv[1:]
    try:
        if args[0] == "prepare":
            await download_build_index()
            exit(0)
        elif args[0] == "translate":
            await translate(args[1])
            exit(0)
    except IndexError:
        pass

    print(f"usage:\t{sys.argv[0]} prepare\n\t{sys.argv[0]} translate <english word>")
    exit(1)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
