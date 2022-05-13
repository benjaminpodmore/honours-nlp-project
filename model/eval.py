from mention_detector import MentionDetector
from mention_trainer import MentionTrainer
from trainer import Trainer
from model import CorefScorer
from loader import train_corpus, val_corpus
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


model = MentionDetector()
mention_trainer = MentionTrainer(model=model, train_corpus=train_corpus, val_corpus=val_corpus, batch_size=15, lr=1e-3)
mention_trainer.load_model('mention_weights/2022-03-08 01-28-09.pth')
preds, targets = mention_trainer.predict(val_corpus[0])
p_cpu = preds.to('cpu').detach()
t_cpu = targets.to('cpu').detach()
print(roc_auc_score(t_cpu, p_cpu))
fpr, tpr, _ = roc_curve(t_cpu, p_cpu)
lw = 2
plt.plot(fpr, tpr, color="orange", lw=lw, linestyle="solid")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.title('ROC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
print(confusion_matrix(t_cpu, p_cpu > 0.03, labels=[1, 0]))


# model = CorefScorer()
# trainer = Trainer(model=model, train_corpus=train_corpus, val_corpus=val_corpus, batch_size=15, lr=1e-3)
# trainer.load_model('model_weights/2022-03-07 17-19-03.pth')
# trainer.train(num_epochs=100, eval_interval=5)
