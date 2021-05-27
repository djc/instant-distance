import asyncio
import json
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
    of less common words. We intercept each line and use this information to build
    an instant-distance index file. We maintain a mapping of english word to embeddings
    in order to convert the translation input to an embedding.
    """
    points = []
    values = []
    word_map = {}

    print("Downloading vector files and building indexes...")
    async with aiohttp.ClientSession() as session:
        for lang in LANGS:
            # Construct a url for each language
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

                            # The first token is the word and the rest
                            # are the embedding
                            value = tokens[0]
                            embedding = [float(p) for p in tokens[1:]]

                            # We only go from english to the other two languages
                            if lang == "en":
                                word_map[value] = embedding
                            else:
                                # We track values here to build the instant-distance index
                                # Every value is prepended with 2 character language code.
                                # This allows us to determine language output later.
                                values.append(lang + value)
                                points.append(embedding)

                            bar.next()

    # Build the instant-distance index and dump it out to a file with .idx suffix
    print("Building index... (this will take a while)")
    hnsw = instant_distance.HnswMap.build(points, values, instant_distance.Config())
    hnsw.dump(BUILT_IDX_PATH)

    # Store the mapping from string to embedding in a .json file
    with open(WORD_MAP_PATH, "w") as f:
        f.write(json.dumps(word_map))


async def translate(word):
    """
    This function relies on the index built in the `download_build_index` function.
    If the data does not yet exist, it will download and build the index.

    The input is expected to be english. A word is first mapped onto an embedding
    from the mapping stored as json. Then we use instant-distance to find the approximate
    nearest neighbors to that point (embedding) in order to translate to other languages.
    """
    data_exists = os.path.isfile(BUILT_IDX_PATH) and os.path.isfile(WORD_MAP_PATH)
    if not data_exists:
        print("Instant Distance index not present. Building...")
        await download_build_index()

    print("Loading indexes from filesystem...")
    with open(WORD_MAP_PATH, "r") as f:
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

    # Print the results
    for result in list(search)[:10]:
        # We know that the first two characters of the value is the language code
        # from when we built the index.
        print(f"Language: {result.value[:2]}, Translation: {result.value[2:]}")


async def main():
    args = sys.argv[1:]
    try:
        if args[0] == "build":
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
