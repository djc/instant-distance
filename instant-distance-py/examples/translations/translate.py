import os
import sys
import instant_distance
import aiohttp
import asyncio
from progress.bar import IncrementalBar
from progress.spinner import Spinner


MAX_LINES = 100_000
LANGS = ("en", "fr", "it")
LANG_REPLACE = "$$lang"
DL_TEMPLATE = f"https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{LANG_REPLACE}.align.vec"
PATH_TEMPLATE = f"./data/wiki.{LANG_REPLACE}.align.trimmed.vec"


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

    print("Downloading vector files and building indexes...")
    async with aiohttp.ClientSession() as session:
        for lang in LANGS:
            # Construct a url for each language and path
            url = DL_TEMPLATE.replace(LANG_REPLACE, lang)
            path = PATH_TEMPLATE.replace(LANG_REPLACE, lang)

            # Ensure the directory and files exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            lineno = 0
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
                                # Every value is prepended with 2 character language code.
                                # This allows us to determine language output later.
                                values.append(lang + tokens[0])
                                vec = tokens[1:]
                                points.append([float(p) for p in vec])

                                # Write out the data to our original .vec files
                                fd.write(linestr)

    # Build the instant-distance index and dump it out to a file with .idx suffix
    print("Building index... (This will take a while)")
    hnsw = instant_distance.HnswMap.build(points, values, id_config)
    hnsw.dump(path.replace(".vec", ".idx"))


async def translate(word):
    data_exists = False
    for path in [PATH_TEMPLATE.replace(LANG_REPLACE, lang) for lang in LANGS]:
        # Ensure .vec and .idx files exist
        data_exists &= os.path.isfile(path)
        data_exists &= os.path.isfile(path.replace(".vec", ".idx"))

    if not data_exists:
        print("Word vector data aren't present. Downloading...")
        await download_build_index()

    print("Loading indexes from filesystem...")


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
