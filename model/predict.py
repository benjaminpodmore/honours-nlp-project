from trainer import Trainer
from model import CorefScorer, CorefScorerGap
from loader import train_corpus, val_corpus
import torch
from spacy.lang.en import English
from text import Document
from utils import bs
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

train_index = 0
m = 500
X_train = np.zeros((m, 2))
y_train = np.zeros((m, 1))

def predict_gap_row(row):
    text = row['Text']
    p_offset = row['Pronoun-offset']
    pronoun = row['Pronoun']

    a_offset = row['A-offset']
    is_a_coref = row['A-coref']
    a_token = row['A']

    b_offset = row['B-offset']
    is_b_coref = row['B-coref']
    b_token = row['B']

    spacy_doc = nlp(text.strip())

    lens = [token.idx for token in spacy_doc]
    pronoun_span = (bs(lens, p_offset) - 1, bs(lens, p_offset) - 1 + len([t.text for t in nlp(pronoun)]) - 1)
    a_span = (bs(lens, a_offset) - 1, bs(lens, a_offset) - 1 + len([t.text for t in nlp(a_token)]) - 1)
    b_span = (bs(lens, b_offset) - 1, bs(lens, b_offset) - 1 + len([t.text for t in nlp(b_token)]) - 1)

    doc = Document(tokens=[token.text for token in spacy_doc], speakers=["-" for token in spacy_doc],
                   corefs=[], raw_text=None, filename=None, sentences=None, )

    spans, probs, g_clusters, r_clusters, p_clusters = trainer.predict_clusters(doc)


    print(text)

    r_correct_find = False
    for cluster in r_clusters:
        print(cluster)
        for (i1, i2) in cluster:
            print(doc.tokens[i1:i2 + 1])
        if a_span in cluster and b_span in cluster:
            if is_a_coref and is_b_coref:
                r_correct_find = True
        elif a_span in cluster and pronoun_span in cluster:
            if is_a_coref:
                r_correct_find = True
        elif b_span in cluster and pronoun_span in cluster:
            if is_b_coref:
                r_correct_find = True
    # if not r_correct_find:
    #     print(index)

    p_correct_find = False
    for cluster in p_clusters:
        print(cluster)
        for (i1, i2) in cluster:
            print(doc.tokens[i1:i2 + 1])
        if a_span in cluster and b_span in cluster:
            if is_a_coref and is_b_coref:
                p_correct_find = True
        elif a_span in cluster and pronoun_span in cluster:
            if is_a_coref:
                p_correct_find = True
        elif b_span in cluster and pronoun_span in cluster:
            if is_b_coref:
                p_correct_find = True



    # if not is_b_coref and not is_a_coref:
    #     print(f"Not either: {index}")


    return r_correct_find, p_correct_find

def predict_gap_row_pronoun(row):
    text = row['Text']
    p_offset = row['Pronoun-offset']
    pronoun = row['Pronoun']

    a_offset = row['A-offset']
    is_a_coref = row['A-coref']
    a_token = row['A']

    b_offset = row['B-offset']
    is_b_coref = row['B-coref']
    b_token = row['B']

    spacy_doc = nlp(text.strip())

    lens = [token.idx for token in spacy_doc]
    pronoun_span = (bs(lens, p_offset) - 1, bs(lens, p_offset) - 1 + len([t.text for t in nlp(pronoun)]) - 1)
    a_span = (bs(lens, a_offset) - 1, bs(lens, a_offset) - 1 + len([t.text for t in nlp(a_token)]) - 1)
    b_span = (bs(lens, b_offset) - 1, bs(lens, b_offset) - 1 + len([t.text for t in nlp(b_token)]) - 1)

    ts = [token.text for token in spacy_doc]
    puncs = [token.text for token in spacy_doc if token.text in ['.', '?', '!']]
    if len(puncs) == 0:
        ts.append(".")

    doc = Document(tokens=ts, speakers=["-" for token in ts],
                   corefs=[], raw_text=None, filename=None, sentences=None, )

    spans, probs, g_clusters, r_clusters, p_clusters = trainer.predict_gap(doc, a_span, b_span, pronoun_span)

    correct = False
    if is_a_coref:
        x = [c for c in r_clusters if a_span in c and pronoun_span in c]
        if len(x) != 0:
            correct = True
    elif is_b_coref:
        x = [c for c in r_clusters if b_span in c and pronoun_span in c]
        if len(x) != 0:
            correct = True
    else:
        if len(r_clusters) == 0:
            correct = True

    return correct


    # pronoun_pred = [(idx, x) for idx, x in enumerate(spans) if x.i1 == pronoun_span[0] and x.i2 == pronoun_span[1]]
    # if len(pronoun_pred) == 0:
    #     return
    # pronoun_pred = pronoun_pred[0]
    # a_preds = [idx for idx, x in enumerate(pronoun_pred[1].yi_idx) if a_span in x]
    # b_preds = [idx for idx, x in enumerate(pronoun_pred[1].yi_idx) if b_span in x]
    # if len(a_preds) == 0 or len(b_preds) == 0:
    #     return
    #
    # prob_a = probs[pronoun_pred[0],a_preds[0]]
    # prob_b = probs[pronoun_pred[0],b_preds[0]]
    #
    # global train_index
    #
    # global X_train
    # global y_train
    # X_train[train_index] = np.array([[prob_a.item(), prob_b.item()]])
    # if not is_a_coref and not is_b_coref:
    #     y_train[train_index] = 0
    # else:
    #     y_train[train_index] = 1
    # # elif is_a_coref:
    # #     y_train[train_index] = 0
    # # elif is_b_coref:
    # #     y_train[train_index] = 1
    #
    # train_index += 1



df = pd.read_csv("../data/gap/gap-dev.tsv", sep='\t')

model = CorefScorer()
trainer = Trainer(model=model, train_corpus=train_corpus, val_corpus=val_corpus, batch_size=15, lr=1e-3)
trainer.load_model('model_weights/2022-03-07 17-19-03.pth')
nlp = English()

doc = nlp("The king fired his servant. He was insane.")
ts = [token.text for token in doc]
doc = Document(tokens=ts, speakers=["-" for token in ts],
               corefs=[], raw_text=None, filename=None, sentences=None, )
spans, probs, g_clusters, r_clusters, p_clusters = trainer.predict_clusters(doc)

r_correct = 0
p_correct = 0
count = 0
# for index, row in df.sample(frac=1).iterrows():
#     if train_index >= m:
#         break
#     print(train_index)
#     predict_gap_row_pronoun(row)
#     # count += 1

for index, row in df.sample(frac=1).iterrows():
    if count >= 50:
        break
    print(count)
    result = predict_gap_row_pronoun(row)
    if result:
        r_correct += 1
    count += 1

# import relevant functions
from sklearn.tree import export_text
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
tree_rules = export_text(clf)
# print the result
print(tree_rules)

# # export the decision rules
# tree_rules = export_text(clf,
#                          feature_names=list(feature_names))
# # print the result
# print(tree_rules)

    # break
# for index, row in df.iloc[872].sample(frac=1).iterrows():
#     r_correct_find, p_correct_find = predict_gap_row(row)
#     break

# predict_gap_row_pronoun(df.iloc[874])

print("halt")
# preds, spans = trainer.predict_text(doc)
# for idx, span in enumerate(spans):
#     n = len(span.yi_idx)
#     span_preds = preds[idx, 0:n+1]
#     coref_id = torch.argmax(span_preds)
#     if coref_id != n:
#         print(coref_id)
#         print(doc.tokens[span.i1:span.i2+1])
#         print(doc.tokens[spans[coref_id.item()].i1:spans[coref_id.item()].i2+1])

