# 🧪 Lab 6: Attention Heatmap Visualization

## 📌 Objective

To visualize how a model focuses on important words using the **Attention Mechanism**.

---

## 🧠 Introduction

Attention allows models to give different importance to different words in a sentence, improving performance and interpretability.

---

## ⚙️ Methodology

### 🔄 Steps Performed

1. Define a sample sentence
2. Assign attention weights
3. Normalize using softmax
4. Visualize using heatmap

---

## 💻 Sample Code

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sentence = ["I", "love", "deep", "learning"]
attention_weights = torch.tensor([0.1, 0.3, 0.4, 0.2])

attention_weights = F.softmax(attention_weights, dim=0).detach().numpy()

sns.heatmap([attention_weights], annot=True, xticklabels=sentence, cmap="Blues")
plt.title("Attention Heatmap")
plt.show()
```

---

## 📊 Expected Output

* Heatmap visualization showing word importance
* Darker color indicates higher importance

---

## 🎯 Conclusion

Attention mechanism helps models focus on relevant parts of input, improving both performance and interpretability.

---

## 🚀 Future Improvements

* Apply attention in real NLP models
* Use transformer-based models
* Visualize attention in longer sentences

---

## 📌 Author

**Om Prakash Kannaujiya**
