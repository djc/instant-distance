import aiohttp
import asyncio
from progress.bar import IncrementalBar

MAX_LINES = 100_000

vector_paths = {
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec": "./data/wiki.en.align.trimmed.vec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec": "./data/wiki.fr.align.trimmed.vec",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.it.align.vec": "./data/wiki.it.align.trimmed.vec"
}

async def download():
    print("Downloading vector files...")
    async with aiohttp.ClientSession() as session:
        for url, path in vector_paths.items():
            lineno = 0
            with IncrementalBar(path.split("/")[-1], max=MAX_LINES) as bar:
                async with session.get(url) as resp:
                    with open(path, 'w') as fd:
                        while True:
                            bar.next()
                            line = await resp.content.readline()
                            if not line:
                                # EOF
                                break

                            lineno += 1

                            # We just use the top 100k embeddings to
                            # save on space and time
                            if lineno >= MAX_LINES:
                                break

                            if lineno == 1:
                                # 100,000 embeddings, 300 dimensions
                                fd.write("100000 300\n")
                            else:
                                fd.write(line.decode("utf-8"))

                            
async def main():
    await download()
    

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
