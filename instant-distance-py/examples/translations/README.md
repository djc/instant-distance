# Translation Example

This example uses the pre-trained aligned word vectors made available by
Facebook Research as part of
[fastText](https://fasttext.cc/docs/en/aligned-vectors.html) to translate
English words to French and Italian.

## Trying it out

Currently, macOS and Python 3.9 is required.

First, install the requirements:

```
pip install -r requirements.txt
```

Run the following command to translate an english word:

```
python translate.py translate hello
```

```
Loading indexes from filesystem...
Language: fr, Translation: hello
Language: it, Translation: hello
Language: it, Translation: ciao
Language: fr, Translation: bonjours
Language: fr, Translation: bonjour
Language: fr, Translation: bonsoir
Language: fr, Translation: !
Language: fr, Translation: salutations
Language: it, Translation: buongiorno
Language: it, Translation: hey
```

The translate command will download the vector data and build an index on the
first run. Subsequent runs will be faster. 

Downloading and building the index takes about **7 minutes** on a MacBook Pro (16-inch, 2019) with 2.4 GHz 8-Core Intel Core i9 and 64GB of RAM and a gigabit internet connection.

If you like, you can rebuild the index at any time:

```
python translate.py build
```
