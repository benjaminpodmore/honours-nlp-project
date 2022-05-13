import io
import os
import re
from fnmatch import fnmatch
from text import Corpus, Document
from utils import flatten


"""
CoNLL processing logic provided by https://github.com/shayneobrien/coreference-resolution
"""

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["%", "*"]


def clean_token(token):
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]

    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')

    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token


def parse_filenames(dirname, pattern="*conll"):
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                yield os.path.join(path, name)


def load_file(filename):
    documents = []
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        raw_text, tokens, text, utts_corefs, utts_speakers, corefs, index = [], [], [], [], [], [], 0
        for line in f:
            raw_text.append(line)
            cols = line.split()

            if len(cols) == 0:
                if text:
                    tokens.extend(text), utts_corefs.extend(corefs), utts_speakers.extend([speaker]*len(text))
                    text, corefs = [], []
                    continue

            elif len(cols) == 2:
                doc = Document(tokens=tokens, corefs=utts_corefs, speakers=utts_speakers,
                               filename=filename, raw_text=raw_text)
                documents.append(doc)
                raw_text, tokens, text, utts_corefs, utts_speakers, index = [], [], [], [], [], 0

            elif len(cols) > 7:
                text.append(clean_token(cols[3]))
                speaker = cols[9]

                if cols[-1] != u'-':
                    coref_expr = cols[-1].split(u'|')
                    for token in coref_expr:

                        match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                        label = match.group(2)

                        if match.group(1) == u'(':
                            corefs.append({'label': label,
                                           'start': index,
                                           'end': None})

                        if match.group(3) == u')':
                            for i in range(len(corefs)-1, -1, -1):
                                if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                    break

                            corefs[i].update({'end': index,
                                              'span': (corefs[i]['start'], index)})

                index += 1
            else:

                continue

    return documents


def read_corpus(dirname):
    files = parse_filenames(dirname=dirname, pattern="*conll")
    return Corpus(flatten([load_file(file) for file in files]))


train_corpus = read_corpus("../data/WikiCoref-CoNLL-master")
val_corpus = read_corpus("../data/val")
