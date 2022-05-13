import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("model/out.csv")

plt.plot(df["MUC F1"], color="navy", label="F1")
plt.plot(df["Coref Recall"], color="orange", label="Recall")
plt.plot(df["Coref Precision"], color="green", label="Precision")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('corefscore.png', dpi=600)
plt.show()
