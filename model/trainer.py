import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
from utils import safe_divide, pad_and_stack, to_cuda, extract_gold_corefs, flatten
from subprocess import Popen, PIPE
import networkx as nx


class Trainer:
    def __init__(self, model, train_corpus, val_corpus, lr=1e-3, batch_size=10):
        self.model = model
        self.train_corpus = list(train_corpus)
        self.val_corpus = val_corpus
        self.batch_size = batch_size

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def train(self, num_epochs, eval_interval=10):
        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified, epoch_f1 = [], [], [], [], []

        for epoch in range(1, num_epochs + 1):
            loss, mentions, recall, precision = self.train_epoch(epoch)

            epoch_loss.append(loss)
            epoch_mentions.append(mentions)
            epoch_corefs.append(recall)
            epoch_identified.append(precision)
            if recall + precision == 0:
                epoch_f1.append(0)
            else:
                epoch_f1.append(2 * (recall * precision) / (recall + precision))

            # TODO check evaluation
            df = pd.DataFrame(list(zip(epoch_loss, epoch_mentions, epoch_corefs, epoch_identified, epoch_f1)),
                              columns=["Loss", "Mentions Found", "Coref Recall", "Coref Precision", "MUC F1"])

            df.to_csv('out.csv')

            if epoch % eval_interval == 0:
                self.save_model(f'{datetime.strftime(datetime.now(), "%Y-%m-%d %H-%M-%S")}')
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        self.model.train()
        batch = random.sample(self.train_corpus, self.batch_size)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for document in tqdm(batch):
            doc = document.truncate(15)

            # loss = self.train_doc(doc)
            #
            loss, mentions_found, total_mentions, \
            corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)
            #
            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                  % (loss, mentions_found, total_mentions,
                     corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

            # # Step the learning rate decrease scheduler
            # # self.scheduler.step()
            #
            # print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
            #       % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
            #          np.mean(epoch_corefs), np.mean(epoch_identified)))

            # summary_writer.add_scalar('Loss/train', loss, epoch)
            print(f"Epoch: {epoch}, loss: {loss}")

        self.scheduler.step()

        return np.mean(epoch_loss), np.mean(epoch_mentions), np.mean(epoch_corefs), np.mean(epoch_identified)

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, total_corefs, \
        gold_mentions, total_mentions = extract_gold_corefs(document)

        translate_dist = document.sentences[0]['acc_length']
        translated_mentions = [(start - translate_dist, end - translate_dist) for (start, end) in gold_mentions if
                               end - start < 10]

        translated_corefs = [
            ((start_i - translate_dist, start_j - translate_dist), (end_i - translate_dist, end_j - translate_dist))
            for ((start_i, start_j), (end_i, end_j)) in gold_corefs]

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document)

        probs, _ = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()

        targets = to_cuda(torch.zeros_like(probs))

        for idx, span in enumerate(spans):
            if (span.i1, span.i2) in translated_mentions:
                mentions_found += 1
                golds = [j for j, link in enumerate(span.yi_idx) if link in translated_corefs]
                if golds:
                    targets[idx, golds] = 1
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    targets[idx, len(span.yi_idx)] = 1

        eps = 1e-8
        loss = -1 * torch.sum(torch.log(F.relu(torch.sum(torch.mul(probs, targets), dim=1)) + eps))
        # Backpropagate

        loss.backward()
        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def save_model(self, savename):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), 'model_weights/' + savename + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath, map_location=torch.device('cpu'))
        self.model.load_state_dict(state)

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)
        probs, _ = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc, clusters

    def predict_clusters(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()
        graph_ranking = nx.Graph()
        graph_pair = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)
        probs, _ = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()
        #
        # # Cluster found coreference links
        # for i, span in enumerate(spans):
        #
        #     # Loss implicitly pushes coref links above 0, rest below 0
        #     found_corefs = [idx
        #                     for idx, _ in enumerate(span.yi_idx)
        #                     if probs[i, idx] > probs[i, len(span.yi_idx)]]
        #
        #     # If we have any
        #     if any(found_corefs):
        #         max_coref_idx = torch.argmax(probs[i:len(span.yi_idx)+1]).item()
        #         link = spans[max_coref_idx]
        #         graph_ranking.add_edge((span.i1, span.i2), (link.i1, link.i2))
        #
        #         last_coref_idx = max(found_corefs)
        #         link = spans[last_coref_idx]
        #         graph_pair.add_edge((span.i1, span.i2), (link.i1, link.i2))
        #
        #         # Add edges between all spans in the cluster
        #         for coref_idx in found_corefs:
        #             link = spans[coref_idx]
        #             graph.add_edge((span.i1, span.i2), (link.i1, link.i2))
        #
        # # Extract clusters as nodes that share an edge
        # clusters = list(nx.connected_components(graph))
        # ranking_clusters = list(nx.connected_components(graph_ranking))
        # pair_clusters = list(nx.connected_components(graph_pair))

        return spans, probs

    def predict_text(self, doc):
        """ Compute loss for a forward pass over a document """

        self.model.eval()


        # Predict coref probabilites for each span in a document
        spans, probs = self.model(doc)

        probs, _ = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()

        return probs, spans

    def predict_gap(self, doc, a_span, b_span, pronoun_span):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()
        graph_ranking = nx.Graph()
        graph_pair = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc, a_span, b_span, pronoun_span)
        probs, _ = pad_and_stack(probs, value=-1e10)
        probs = probs.squeeze()

        if probs[2, 0].item() > probs[2, 2].item() and probs[2, 1].item() > probs[2, 2].item():
            if abs(probs[2, 1].item() - probs[2, 0]) < 0.15:
                print("wait")

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):
                max_coref_idx = torch.argmax(probs[i:len(span.yi_idx)+1]).item()
                link = spans[max_coref_idx]
                graph_ranking.add_edge((span.i1, span.i2), (link.i1, link.i2))

                last_coref_idx = max(found_corefs)
                link = spans[last_coref_idx]
                graph_pair.add_edge((span.i1, span.i2), (link.i1, link.i2))

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))
        ranking_clusters = list(nx.connected_components(graph_ranking))
        pair_clusters = list(nx.connected_components(graph_pair))

        return spans, probs, clusters, ranking_clusters, pair_clusters

    def evaluate(self, val_corpus, eval_script='../eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in tqdm(val_corpus)]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        # results = str(stdout).split('TOTALS')[-1]
        results = str(stdout).replace("\\n", "\n")

        # Write the results out for later viewing
        with open('../preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = '../preds/golds.txt', '../preds/predictions.txt'
        if not os.path.exists('../preds/'):
            os.makedirs('../preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            for doc in val_corpus:

                current_idx = 0

                for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()
                        tokens[-1] = doc.tags[current_idx]

                        # Increment by 1 so tags are still aligned
                        current_idx += 1

                        # Rewrite it back out
                        f.write('\t'.join(tokens))
                    f.write('\n')

        return golds_file, preds_file
