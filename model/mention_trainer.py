import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
from utils import safe_divide, pad_and_stack, to_cuda, extract_gold_corefs, compute_idx_spans


class MentionTrainer:
    def __init__(self, model, train_corpus, val_corpus, lr=1e-3, batch_size=10):
        self.model = model
        self.train_corpus = list(train_corpus)
        self.val_corpus = list(val_corpus)
        self.batch_size = batch_size

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def train(self, num_epochs, eval_interval=10):
        epoch_loss, epoch_mentions, mentions_identified, epoch_f1 = [], [], [], []

        for epoch in range(1, num_epochs + 1):
            loss, mentions, precision = self.train_epoch(epoch)
            epoch_loss.append(loss)
            epoch_mentions.append(mentions)
            mentions_identified.append(precision)
            if mentions + precision == 0:
                epoch_f1.append(0)
            else:
                epoch_f1.append(2 * (mentions * precision) / (mentions + precision))

            df = pd.DataFrame(list(zip(epoch_loss, epoch_mentions, mentions_identified, epoch_f1)),
                              columns=["Loss", "Mentions Found", "Mention Precision", "MUC F1"])

            df.to_csv('mention_out.csv')

            if epoch % eval_interval == 0:
                self.save_model(f'{datetime.strftime(datetime.now(), "%Y-%m-%d %H-%M-%S")}')

    def train_epoch(self, epoch):
        self.model.train()
        batch = random.sample(self.train_corpus, self.batch_size)

        epoch_loss, epoch_mentions, mentions_identified = [], [], []

        for document in tqdm(batch):
            doc = document.truncate(15)
            # doc = document
            loss, mentions_found, correct_mentions, total_mentions, = self.train_mentions(doc)
            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions Found: %d/%d | Mention precision: %d/%d' \
                  % (loss, mentions_found, total_mentions, correct_mentions, total_mentions))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            mentions_identified.append(safe_divide(correct_mentions, total_mentions))

            print(f"Epoch: {epoch}, loss: {loss}")

        self.scheduler.step()

        return np.mean(epoch_loss), np.mean(epoch_mentions), np.mean(mentions_identified)

    def train_mentions(self, document):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, total_corefs, \
        gold_mentions, total_mentions = extract_gold_corefs(document)

        translate_dist = document.sentences[0]['acc_length']
        translated_mentions = [(start - translate_dist, end - translate_dist) for (start, end) in gold_mentions if
                               end - start < 10]

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        correct_mentions = 0
        mentions_found = 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document)

        preds = nn.Sigmoid()(probs)

        targets = to_cuda(torch.zeros_like(probs)).detach()

        for idx, i in enumerate(compute_idx_spans(document.sents, L=10)):
            mention = (i[0], i[-1])
            if mention in translated_mentions:
                targets[idx] = 1.

        for idx, span in enumerate(spans):
            if (span.i1, span.i2) in translated_mentions:
                mentions_found += 1
                correct_mentions += sum(preds[idx] > 0.5).item()

        loss = nn.BCELoss()(preds, targets)
        # loss = -1 * torch.sum(torch.mul(targets, torch.log(preds)) + torch.mul(1 - targets, torch.log(1 - preds)))

        # Backpropagate
        loss.backward()
        # Step the optimizer
        self.optimizer.step()

        return loss.item(), mentions_found, correct_mentions, total_mentions


    def predict(self, doc):
        # Extract gold coreference links
        gold_corefs, total_corefs, \
        gold_mentions, total_mentions = extract_gold_corefs(doc)

        translate_dist = doc.sentences[0]['acc_length']
        translated_mentions = [(start - translate_dist, end - translate_dist) for (start, end) in gold_mentions if
                               end - start < 10]

        # Init metrics
        correct_mentions = 0
        mentions_found = 0

        self.model.eval()
        # Predict coref probabilites for each span in a document
        spans, probs = self.model(doc)

        preds = nn.Sigmoid()(probs)

        targets = to_cuda(torch.zeros_like(probs)).detach()

        for idx, i in enumerate(compute_idx_spans(doc.sents, L=10)):
            mention = (i[0], i[-1])
            if mention in translated_mentions:
                targets[idx] = 1.

        return preds, targets


    def save_model(self, savename):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), 'mention_weights/' + savename + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath, map_location=torch.device('cpu'))
        self.model.load_state_dict(state)
