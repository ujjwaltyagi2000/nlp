# **Word Embeddings Explained**

Word embeddings have become integral to tasks such as text classification, sentiment analysis, machine translation and more.

Traditional methods **_(frequency based embeddings)_** of representing words in a way that machines can understand, such as one-hot encoding, represent each word as a sparse vector with a dimension equal to the size of the vocabulary. Here, only one element of the vector is "hot" (set to 1) to indicate the presence of that word. While simple, this approach suffers from the curse of dimensionality, lacks semantic information and doesn't capture relationships between words.

**_Prediction-based embeddings_**, on the other hand, are dense vectors with continuous values that are trained using machine learning techniques, often based on neural networks. The idea is to learn representations that encode semantic meaning and relationships between words. Word embeddings are trained by exposing a model to a large amount of text data and adjusting the vector representations based on the context in which words appear.

Prediction-based embeddings can differentiate between synonyms and handle polysemy (multiple meanings of a word) more effectively. The vector space properties of prediction-based embeddings enable tasks like measuring word similarity and solving analogies. Prediction-based embeddings can also generalize well to unseen words or contexts, making them robust in handling out-of-vocabulary terms.

One popular method for training prediction-based embeddings is **Word2Vec**, which uses a neural network to predict the surrounding words of a target word in a given context. Another widely used approach is GloVe (Global Vectors for Word Representation), which leverages global statistics to create embeddings.

Below are the notes for these techniques. For more info on these embeddings and how they work under the hood, go [here](https://www.ibm.com/think/topics/word-embeddings) for a general overview and [here](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/#h-introduction) for a more elaborate explaination.

### **How Word Embeddings Work**

Each word is represented as a **dense, real-valued vector** in a high-dimensional space. The key idea is that **words with similar meanings will have similar vector representations**.

For example:

- **King** and **Queen** will have similar embeddings.
- **Apple** and **Orange** will be closer than **Apple** and **Car**.

Word embeddings are trained using **context-based learning**, where the model learns relationships between words from a large corpus.

---

### **Popular Word Embedding Techniques**

1. **Word2Vec (by Google)**
   - Uses **Skip-gram** and **CBOW (Continuous Bag of Words)** to predict word relationships.
   - Example: `King - Man + Woman ≈ Queen`
2. **GloVe (Global Vectors for Word Representation - by Stanford)**

   - Learns embeddings based on word co-occurrence in a large corpus.
   - Good for capturing global context.

3. **FastText (by Facebook)**

   - Improves Word2Vec by considering subword information, making it better for rare and misspelled words.

4. **BERT, GPT (Transformer-based embeddings)**
   - Contextual embeddings that capture the meaning of words **based on their usage in a sentence**.
   - Example: The word "bank" in "river bank" vs. "money bank" will have different embeddings.

---

### **Advantages of Word Embeddings**

✅ **Captures semantic relationships** between words.  
✅ **Reduces dimensionality** compared to one-hot encoding.  
✅ **Handles synonyms better** than BoW and TF-IDF.  
✅ **Improves generalization** in NLP models.

---

---

# **Word2Vec**

Developed by Google in 2013, Word2Vec is fundamentally different from traditional encoding methods like **One-Hot Encoding, Bag of Words (BoW), and TF-IDF** because it captures the **semantic meaning** of words rather than just their occurrence patterns.

| **Method**             | **Representation Type**   | **Captures Meaning?** | **Handles Context?** | **Sparse or Dense?** | **Scalability**    |
| ---------------------- | ------------------------- | --------------------- | -------------------- | -------------------- | ------------------ |
| **One-Hot Encoding**   | Binary Vector             | ❌ No                 | ❌ No                | **Sparse**           | 🚫 Not Scalable    |
| **Bag of Words (BoW)** | Frequency-based Vector    | ❌ No                 | ❌ No                | **Sparse**           | 🚫 Not Scalable    |
| **TF-IDF**             | Weighted Frequency Vector | ❌ No                 | ❌ No                | **Sparse**           | ✅ Better than BoW |
| **Word2Vec**           | Dense Embedding Vector    | ✅ Yes                | ✅ Yes               | **Dense**            | ✅ Highly Scalable |

---

### **Disadvantages of One-Hot, BoW, and TF-IDF That Word2Vec Fixes**

#### **1️⃣ Dimensionality Problem**

- **One-Hot Encoding & BoW** create **huge sparse vectors**, making computations inefficient.
- **Word2Vec** produces **compact, dense vectors**, reducing memory usage.

#### **2️⃣ Lack of Meaning & Context**

- **One-Hot, BoW, and TF-IDF** do **not** capture relationships between words.
- **Word2Vec** places similar words **closer in vector space** (e.g., "king" and "queen" will have similar embeddings).

#### **3️⃣ No Understanding of Word Similarity**

- In **One-Hot, BoW, and TF-IDF**, words are independent; no relation exists between "cat" and "dog."
- **Word2Vec** learns that similar words have **similar embeddings** by training on a large corpus.

#### **4️⃣ Fixed Vocabulary Issue**

- **One-Hot & BoW require fixed vocabulary**, so new words are **ignored**.
- **Word2Vec** can generalize and find similarities for unseen words using pre-trained embeddings.

---

### **Two Models of Word2Vec**

Word2Vec has **two architectures**:

- **Continuous Bag of Words (CBOW)**
- **Skip-gram Model**

Both models use **a shallow neural network** with **one hidden layer** and are trained using a **large text corpus**.

---

## **1️⃣ Continuous Bag of Words (CBOW)**

CBOW tries to **predict a missing target word** using its surrounding **context words**. It works like **fill-in-the-blank**.

👉 **Example:**  
Let's consider the sentence:  
💡 _"The cat sat on the **mat**."_

Here, we take **context words** (surrounding words) and try to predict the **target word** (the missing word in the middle).

| Context Words      | Target Word |
| ------------------ | ----------- |
| The, cat, on, the  | **sat**     |
| cat, sat, the, mat | **on**      |
| sat, on, the       | **mat**     |

### **CBOW Architecture**

1. **Input Layer**: Accepts the surrounding words (context words).
2. **Hidden Layer**: A **projection layer** with weights that learn relationships between words.
3. **Output Layer**: Uses **softmax** to predict the probability of a word being the missing word.

🟢 **Advantages of CBOW**  
✔ Faster to train compared to Skip-gram.  
✔ Works well with **frequent words**.  
✔ Requires **less data** than Skip-gram.

🔴 **Disadvantages of CBOW**  
❌ Does **not** work well with rare words.  
❌ Averages context words, losing word order information.

---

## **2️⃣ Skip-gram Model**

Skip-gram **does the opposite** of CBOW. Instead of predicting the target word from context words, it **predicts surrounding context words given a target word**.

👉 **Example:**  
💡 _"The cat sat on the mat."_

If the model picks the **target word** "**sat**", it tries to predict the nearby words:

| Target Word | Context Words     |
| ----------- | ----------------- |
| **sat**     | The, cat, on, the |

### **Skip-gram Architecture**

1. **Input Layer**: Takes a single word (the target word).
2. **Hidden Layer**: Maps the word to a dense **vector representation**.
3. **Output Layer**: Predicts the probability of surrounding words appearing in context.

🟢 **Advantages of Skip-gram**  
✔ **Performs well on rare words**.  
✔ Works better for **larger datasets**.  
✔ More accurate for learning word relationships.

🔴 **Disadvantages of Skip-gram**  
❌ **Slower** to train compared to CBOW.  
❌ Needs **more training data** to generalize well.

---

## **How the models are trained**

Given a sequence of words in a sentence, the CBOW model takes a fixed number of context words (words surrounding the target word) as input. Each context word is represented as an embedding (vector) through a shared embedding layer. These embeddings are learned during the training process.

The individual context word embeddings are aggregated, typically by summing or averaging them. This aggregated representation serves as the input to the next layer.

The aggregated representation is then used to predict the target word using a softmax activation function. The model is trained to minimize the difference between its predicted probability distribution over the vocabulary and the actual distribution (one-hot encoded representation) for the target word.

The CBOW model is trained by adjusting the weights of the embedding layer based on its ability to predict the target word accurately.

The Continuous Skip-gram model uses training data to predict the context words based on the target word's embedding. Specifically, it outputs a probability distribution over the vocabulary, indicating the likelihood of each word being in the context given the target word.

The training objective is to maximize the likelihood of the actual context words given the target word. This involves adjusting the weights of the embedding layer to minimize the difference between the predicted probabilities and the actual distribution of context words. The model also allows for a flexible context window size. It can be adjusted based on the specific requirements of the task, allowing users to capture both local and global context relationships.

The Skip-gram model is essentially "skipping" from the target word to predict its context, which makes it particularly effective in capturing semantic relationships and similarities between words.

---

## **CBOW vs. Skip-gram – A Comparison**

| Feature                       | CBOW                                     | Skip-gram                                  |
| ----------------------------- | ---------------------------------------- | ------------------------------------------ |
| **Prediction Type**           | Predicts target word using context words | Predicts context words using a target word |
| **Training Speed**            | **Faster**                               | **Slower**                                 |
| **Performance on Rare Words** | Not good                                 | **Better**                                 |
| **Data Requirement**          | Works well with **less data**            | Needs **more data**                        |
| **Best Use Case**             | **Frequent words, smaller datasets**     | **Rare words, larger datasets**            |

---

## **Which One to Use?**

- **If your dataset is small and you want a faster model** → **CBOW is better.**
- **If your dataset is large and you need high-quality embeddings** → **Skip-gram is better.**
