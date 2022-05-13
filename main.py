import spacy
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import neuralcoref

nlp = spacy.load("en_core_web_sm")


def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)  # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")  # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)  # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])",
                           r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]  # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)


nlp.tokenizer = custom_tokenizer(nlp)
nlp.tokenizer.add_special_case("90m", [{ORTH: "90m"}])

from xml.dom import minidom

file = minidom.parse('data/WikiCoref/Annotation/Aberfoyle,_Stirling/Markables/Aberfoyle, Stirling_coref_level.xml')

items = []
models = file.getElementsByTagName('markable')
for model in models:
    item = {}
    for i in range(len(model.attributes)):
        item[model.attributes.item(i).nodeName] = model.attributes.item(i).value
        # print(model.attributes.item(i).nodeName, model.attributes.item(i).value)
    items.append(item)

doc = minidom.parse("data/WikiCoref/Annotation/Aberfoyle,_Stirling/Basedata/Aberfoyle, Stirling_words.xml")

words = doc.getElementsByTagName("word")


def getWord(word, offset=0):
    return {
        'id': f"{word.attributes.item(0).value.split('_')[0]}_{int(word.attributes.item(0).value.split('_')[1]) + offset}",
        "word": word.firstChild.data}


wordList = []
offset = 0
for word in words:
    #     print(getWord(word))
    temp = getWord(word, offset)['word']
    #     if "-" in temp and temp != '-LRB-' and temp != '-RRB-':
    #             print(temp)
    #             pos = list(findall('-',temp))
    #             if len(pos) == 1:
    #                 wordList.append(getWordSplit(word,offset,0))
    #                 wordList.append(getHyphen(word,offset + 1))
    #                 wordList.append(getWordSplit(word,offset + y2,1))
    #                 offset += 2
    #     else:
    wordList.append(getWord(word, offset))

s = ''
for entry in wordList:
    s += entry["word"]
    s += ' '

doc = nlp(s)
print(doc[188])

for l in wordList:
    i = int(l["id"].split("_")[1])
    print("spacy", doc[i - 1])
    print("mmax", l["word"])
    if str(doc[i - 1]) != l["word"]:
        break
