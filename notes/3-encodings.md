# **Types of Encodings in NLP & Their Practical Uses**

Encoding is crucial in NLP because computers cannot understand text directly. Instead, we must **convert words, sentences, or even entire documents into numerical representations (vectors/encodings)**. These representations allow machine learning models to process and analyze text efficiently.

---

---

## **1. One-Hot Encoding (OHE) in NLP**

---

One-hot encoding is a way to represent words (or categorical data) as **binary vectors**. Each word is assigned a **unique index**, and its vector has **1 at its position** while all other positions are **0**.

For example, if we have the words:  
**["cat", "dog", "fish"]**, we can represent them as:

| Word     | cat | dog | fish |
| -------- | --- | --- | ---- |
| **cat**  | 1   | 0   | 0    |
| **dog**  | 0   | 1   | 0    |
| **fish** | 0   | 0   | 1    |

### **📌 Where is One-Hot Encoding Used?**

🔹 **Simple Text Classification Tasks**:

- Spam detection (classifying emails as spam/not spam).
- Sentiment analysis (positive/negative reviews).
- Fake news detection.

🔹 **Text Representation in Small Datasets**:

- If the vocabulary is small, OHE can be used as an input to simple models.

🔹 **Feature Engineering for Traditional ML Models**:

- Some ML algorithms like **Naive Bayes** and **Decision Trees** work well with OHE.

### **📌 Advantages of One-Hot Encoding**

**1️⃣ Simplicity**

- **Easy to implement** and works well for small vocabularies.
- No need for complex training or pre-trained models.

**2️⃣ Works Well for Rule-Based Models**

- If you are using **decision trees, rule-based models, or counting-based methods**, one-hot encoding can be effective.

**3️⃣Useful for Small Datasets**

- If you have **a small vocabulary**, one-hot encoding can work without the need for complex NLP techniques.

### **📌 Disadvantages of One-Hot Encoding**

**1️⃣ High Dimensionality (Curse of Dimensionality)**

- If you have **100,000 words**, OHE creates **100,000-dimensional vectors**!
- This leads to **huge memory usage** and makes computations inefficient.

🔹 **Example**:  
If you apply OHE to a dataset with **1 million unique words**, you get **a 1,000,000 × 1,000,000 sparse matrix** (mostly zeros).

**2️⃣ No Semantic Meaning or Context**

- **OHE treats all words as independent** and doesn’t capture relationships (very important).
- "king" and "queen" have totally different vectors, even though they are related.
- On applying metrics like cosine similarity on these vectors, all vectors are mostly found equidistant from one another therefore deriving relationships becomes really hard.

🔹 **Example**:

- `"good"` and `"excellent"` should be similar, but in OHE, they are completely different.

✅ **Better Alternative**: **Word Embeddings (Word2Vec, GloVe, BERT)**  
These **capture relationships** between words and improve NLP tasks.

**3️⃣ Sparsity (Inefficient Computation)**

- Most values in OHE vectors are **zeros**, making it a **sparse representation**.
- Sparse matrices slow down training and increase memory usage.
- Often results in overfitting.

✅ **Solution**:

- Use **TF-IDF** (weighs words based on importance).
- Use **Word Embeddings** (more compact and meaningful).

### **📌 When Should You Avoid One-Hot Encoding?**

❌ **Large Vocabulary Size** → Use embeddings instead.  
❌ **When Context Matters** → OHE does not capture word relationships.  
❌ **For Deep Learning** → Models like LSTMs, CNNs, or Transformers perform better with embeddings.

### **📌 Conclusion: When to Use OHE?**

✅ **For small vocabularies (e.g., up to a few hundred words).**  
✅ **For simple classification tasks where relationships between words don’t matter.**  
✅ **When using traditional ML models like Decision Trees or Naive Bayes.**

---

---

## **2. Bag of Words (BoW) Encoding in NLP**

Bag of Words (BoW) is a simple way to **convert text into numerical features** for machine learning models. It represents **how frequently words appear** in a document **without considering their order** or context.

### **📌 How BoW Works**

1️⃣ **Create a Vocabulary** (a set of unique words in the dataset).  
2️⃣ **Count Word Occurrences** in each document.  
3️⃣ **Convert into a Vector Representation.**

**Example**:  
Consider these two sentences:  
📌 **Sentence 1:** "I love NLP and Machine Learning."  
📌 **Sentence 2:** "I love Deep Learning and NLP."

🔹 **Step 1: Create a Vocabulary**  
Unique words in both sentences →  
📌 **["I", "love", "NLP", "and", "Machine", "Learning", "Deep"]**

🔹 **Step 2: Create Word Frequency Vectors**

|                | I   | love | NLP | and | Machine | Learning | Deep |
| -------------- | --- | ---- | --- | --- | ------- | -------- | ---- |
| **Sentence 1** | 1   | 1    | 1   | 1   | 1       | 1        | 0    |
| **Sentence 2** | 1   | 1    | 1   | 1   | 0       | 1        | 1    |

Each row is a **numerical representation of a sentence** based on word counts.

**Example where word frequency is greater than one**

Consider this small corpus:

📌 **Sentence 1:** "I love NLP and NLP loves me"
📌 **Sentence 2:** "NLP is amazing and I love it"
📌 **Sentence 3:** "Deep learning and NLP are the future"

**Vocabulary from BoW:** ['I', 'love', 'NLP', 'and', 'loves', 'me', 'is', 'amazing', 'it', 'Deep', 'learning', 'are', 'the', 'future']`

**BoW Representation (word counts instead of binary 0/1):**

```
[
    [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # First sentence (NLP appears twice)
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Second sentence
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # Third sentence
]
```

- Here, **'NLP' appears twice** in the first sentence, so its count is `2`, unlike the previous binary representation.

### **Types of BoW Encoding**

1. **Binary BoW (Boolean Representation)**
   - Marks presence (`1`) or absence (`0`) of words.
   - Example: `"I love NLP"` → `[1, 1, 1, 0, 0]`
2. **Count BoW (Raw Frequency Representation)**
   - Stores the actual frequency of each word.
   - Example: `"I love NLP and NLP"` → `[1, 1, 2, 1, 0]` (NLP appears twice)

### **📌 Where is BoW Used?**

✅ **Text Classification** (Spam detection, sentiment analysis).  
✅ **Topic Modeling** (Identifying topics in a document).  
✅ **Information Retrieval** (Search engines).

### **📌 Advantages of BoW**

**1️⃣ Simple and Fast**

- No need for **pre-trained embeddings** or **deep learning**.
- Works well with traditional **ML models like Naïve Bayes, Logistic Regression, and SVM**.

**2️⃣ Captures Word Frequency**

- Words that appear more often get higher importance, which can be useful for **text classification tasks** like spam detection and sentiment analysis.

**3️⃣ Works Well When Context is Not Important**

- In some tasks like **topic classification**, knowing word frequency is enough—word order isn’t needed.

**4️⃣ Compatible with Traditional Machine Learning**

- Unlike **word embeddings**, BoW can be used with classical ML models without needing deep learning as it has fixed size inputs.

### **📌 Disadvantages of BoW**

**1️⃣ Ignores Word Order**

- "I love NLP" and "NLP love I" have the **same representation** even though they have different meanings.
- This makes BoW **bad for tasks like sentence generation or language translation**.

**2️⃣ Out-of-Vocabulary Words**

- Unable to handle words not present in the vocabulary, posing challenges for unseen terms. This limitation becomes pronounced in dynamic or evolving language environments.

**3️⃣ Doesn't Capture Meaning or Context**

- The words "good" and "excellent" are similar, but BoW treats them **as separate words**.
- It **doesn’t capture synonyms** or relationships between words.

**4️⃣ Difficulty with Synonyms:**

- Struggles with distinguishing between words with the same spelling but different meanings (homonyms). The model may treat them as the same entity.

**5️⃣ Difficulty with Homonyms:**

- Treats synonyms as distinct words, not recognizing their semantic similarity. This limitation impacts tasks that require understanding the underlying meaning of text.

### **📌 When Should You Use BoW?**

✅ **For Simple Text Classification Tasks** (Spam detection, sentiment analysis).  
✅ **When Training Traditional ML Models** (Logistic Regression, SVM).  
✅ **If Your Dataset is Small** (BoW struggles with large vocabularies).

### **📌 Conclusion**

✔ **BoW is easy to implement and effective for basic tasks**.  
❌ **Does not capture meaning, word order, or relationships**.  
❌ **Can face problems with out of vocabulary words, homonyms, synonyms**.

---

---

### **Disadvantages of One-Hot Encoding that Bag of Words Fixes**

1. **Lack of Frequency Information**

   - **One-Hot Encoding:** Only indicates whether a word is present (1) or absent (0), without considering how many times the word appears.
   - **BoW Fix:** BoW (in count form) captures word frequency, which helps in understanding the importance of words in a document.

2. **Scalability Issues with Large Vocabularies**

   - **One-Hot Encoding:** Creates **very high-dimensional** sparse vectors (mostly zeros), as each unique word gets its own dimension.
   - **BoW Fix:** Still produces high-dimensional vectors, but by using **count-based features or TF-IDF weighting**, it reduces the total memory needed compared to one-hot encoding.

3. **No Context or Semantic Meaning**

   - **One-Hot Encoding:** Treats words as independent entities, completely ignoring context. "King" and "Queen" have no relation in a one-hot vector.
   - **BoW Fix:** While BoW still lacks deep semantic understanding, **word frequency patterns** help capture some level of document similarity.

4. **Inefficiency for Large Datasets**
   - **One-Hot Encoding:** With large vocabularies, vector dimensions **explode**, making it computationally expensive.
   - **BoW Fix:** Can be combined with **dimensionality reduction techniques (e.g., TF-IDF, stop-word removal, n-grams, LSA)** to improve efficiency.

### **Summary**

| Feature           | One-Hot Encoding   | Bag of Words (BoW)             |
| ----------------- | ------------------ | ------------------------------ |
| Word Frequency    | ❌ Not captured    | ✅ Captured                    |
| Vector Size       | ❌ Extremely large | ⚠️ Large but manageable        |
| Context Awareness | ❌ None            | ❌ None (still a drawback)     |
| Computation       | ❌ Inefficient     | ✅ More efficient than One-Hot |

---

---

### **N-Grams in NLP**

An **N-Gram** is a sequence of **N** consecutive words (or characters) from a given text. It is used to capture context and dependencies between words.

#### **Types of N-Grams**

- **Unigram (N = 1)** → Single words
  - _Example:_ `"I love NLP"` → `['I', 'love', 'NLP']`
- **Bigram (N = 2)** → Pairs of consecutive words
  - _Example:_ `"I love NLP"` → `['I love', 'love NLP']`
- **Trigram (N = 3)** → Three-word sequences
  - _Example:_ `"I love NLP"` → `['I love NLP']`
- **Higher-order N-Grams (N > 3)** → Longer sequences
  - _Example (N=4):_ `"I love NLP models"` → `['I love NLP models']`

---

### **Why Use N-Grams?**

✅ **Captures context:** Unlike BoW, which ignores word order, n-grams capture local dependencies between words.  
✅ **Useful for text prediction:** Helps in **autocorrect**, **autocomplete**, and **language modeling** (e.g., **Google search predictions**).  
✅ **Better representation for short texts:** In short phrases, word co-occurrence is critical for meaning.

---

### **Example: How N-Grams Help in NLP?**

Let's say we want to predict the next word:

💬 **Input Sentence:** `"I want to eat"`  
💡 **Bigram Model Suggestion:** `"I want to eat pizza"`  
💡 **Trigram Model Suggestion:** `"I want to eat delicious food"`

The higher the N, the **better the context capture**, but also the **higher the computational cost**.

---

---

### **TF-IDF (Term Frequency - Inverse Document Frequency)**

TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It is widely used in **text mining, search engines, and NLP tasks** to find the most relevant words in a document.

---

### **Formula Breakdown**

TF-IDF is the product of two components:

**1. Term Frequency (TF)**

Measures how often a term appears in a document.

$$
TF = \frac{\text{Number of times a term appears in a document}}{\text{Total number of terms in the document}}
$$

- **Example**: If the word "machine" appears **3 times** in a document with **100 words**, then:
  $$
  TF = \frac{3}{100} = 0.03
  $$

**2. Inverse Document Frequency (IDF)**

Measures how important a word is by reducing the weight of commonly used words (e.g., "the", "is").

$$
IDF = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents containing the term}} + 1 \right)
$$

- **Example**: If we have **10,000 documents** and the word "machine" appears in **1,000** of them:
  $$
  IDF = \log \left( \frac{10,000}{1,000} + 1 \right) = \log(11) \approx 2.4
  $$

**3. TF-IDF Calculation**

$$
TF-IDF = TF \times IDF
$$

- If **TF = 0.03** and **IDF = 2.4**, then:
  $$
  TF-IDF = 0.03 \times 2.4 = 0.072
  $$
- Higher values indicate that the word is **important** in the document but **rare** across the corpus.

---

### **Advantages of TF-IDF**

✅ Helps filter out **common words** that do not carry meaning.  
✅ **Gives weight to important words** that appear in fewer documents.  
✅ Works well for **information retrieval (search engines)**.
✅ Fixed Size inputs (equal to vocabulary) is optimal for training ML models.

### **Limitations of TF-IDF**

❌ **Ignores word order** (e.g., "New York" vs. "York New").  
❌ **Does not capture meaning** (e.g., synonyms "car" and "automobile" are treated separately).  
❌ **Sensitive to long documents** (frequent words dominate).
❌ Sparsity still exists.
❌ Out of Vocabulary

---

### **Where is TF-IDF Used?**

🔹 **Search engines** (Google ranks pages based on TF-IDF).  
🔹 **Text classification** (spam detection, sentiment analysis).  
🔹 **Keyword extraction** (summarization and topic modeling).  
🔹 **Document similarity** (finding similar articles).

---

### **Comparison: Bag of Words (BoW) vs. TF-IDF**

| Feature              | **Bag of Words (BoW)**                            | **TF-IDF (Term Frequency-Inverse Document Frequency)**                                      |
| -------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Definition**       | Counts the frequency of words in a document.      | Weighs words based on their importance in a document relative to the entire corpus.         |
| **Weighting**        | All words are treated equally.                    | Words are weighted based on their frequency in the document and their rarity in the corpus. |
| **Common Words**     | Common words (e.g., "the", "is") get high values. | Common words are downweighted using IDF.                                                    |
| **Rare Words**       | Rare words are not distinguished.                 | Rare words get a higher weight, improving importance.                                       |
| **Interpretability** | Simple and easy to understand.                    | More complex due to logarithmic scaling.                                                    |
| **Use Cases**        | Basic text classification, spam detection.        | Information retrieval, search engines, keyword extraction.                                  |
| **Sparsity**         | Highly sparse due to large vocabulary size.       | Still sparse, but weights help reduce noise.                                                |

---

### **Is TF-IDF Better Than BoW?**

✅ **TF-IDF is generally better than BoW** because:

1. **It reduces the impact of common words**, making it more effective for understanding important terms.
2. **It highlights rare but meaningful words**, improving classification and search relevance.
3. **It prevents bias from high-frequency words** that may not carry much meaning.

However, **BoW can still be useful** for simple tasks like text classification when you don’t need term weighting.
