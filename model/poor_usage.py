from trainer import Trainer
from model import CorefScorer
from loader import train_corpus, val_corpus
import torch

model = CorefScorer()
trainer = Trainer(model=model, train_corpus=train_corpus, val_corpus=val_corpus, batch_size=15, lr=1e-3)
trainer.load_model('model_weights/2022-03-07 17-19-03.pth')
import pandas as pd
import numpy as np
from spacy.lang.en import English
from text import Document
nlp = English()
df = pd.read_excel("../data/gap/MY_DATASET.xlsx")
n = len(df) - 4

X_train = np.zeros((n,3))
y_train = np.zeros(n)

# for index, row in df.sample(frac=1).iterrows():
for index, row in df.iloc[0:n].iterrows():
    spacy_doc = nlp(row['Text'])
    doc = Document(tokens=[token.text for token in spacy_doc], speakers=["-" for token in spacy_doc],
               corefs=[], raw_text=None, filename=None, sentences=None)
    # spans, probs, g_clusters, r_clusters, p_clusters = trainer.predict_clusters(doc)
    spans, probs  = trainer.predict_clusters(doc)
    span_tokens = [(idx, doc.tokens[span.i1:span.i2 + 1]) for idx, span in enumerate(spans)]
    a_span_tokens = [span_token for span_token in span_tokens if " ".join(span_token[1]) == row["A"]][0]
    b_span_tokens = [span_token for span_token in span_tokens if " ".join(span_token[1]) == row["B"]][0]
    p_span_tokens = [span_token for span_token in span_tokens if " ".join(span_token[1]) == row["Pronoun"]][0]

    ap_prob = probs[p_span_tokens[0]-1, a_span_tokens[0]]
    bp_prob = probs[p_span_tokens[0]-1, b_span_tokens[0]]
    diff_prob = abs(ap_prob - bp_prob)
    print(row["Text"])
    print("A Prob", ap_prob.item())
    print("B Prob", bp_prob.item())
    print("Prob Diff", diff_prob.item())
    print(probs[p_span_tokens[0]-1])
    print(torch.argmax(probs[p_span_tokens[0]-1]).item())
    print()

    if row["IsPoor"]:
        y_train[index] = 1

    X_train[index, 0] = diff_prob
    X_train[index, 1] = ap_prob
    X_train[index, 2] = bp_prob