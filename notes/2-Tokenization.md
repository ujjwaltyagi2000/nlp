# Tokenization in NLP

### **Definition**

Tokenization in natural language processing (NLP) is a technique that involves dividing a sentence or phrase into smaller units known as tokens. These tokens can encompass words, dates, punctuation marks, or even fragments of words.

Tokenization involves using a tokenizer to segment unstructured data and natural language text into distinct chunks of information, treating them as different elements. The tokens within a document can be used as vector, transforming an unstructured text document into a numerical data structure suitable for machine learning.

### **Terminologies**

1. **Corpus (Plural: Corpora)**  
   A **corpus** is a large collection of text used for training or analyzing NLP models. It can be domain-specific (e.g., medical corpus, legal corpus) or general-purpose (e.g., Wikipedia corpus).

   - Example: A dataset of all Wikipedia articles in English.

2. **Document**  
   A **document** refers to a single unit of text within a corpus. It could be a sentence, paragraph, webpage, article, or any structured piece of text.

   - Example: A single news article in a dataset of news reports.

3. **Sentence**  
   A **sentence** is a structured sequence of words conveying a complete thought. Sentence tokenization (segmentation) breaks text into individual sentences.

   - Example: _"NLP is interesting. I love learning about it."_ → Two sentences.

4. **Token**  
   A **token** is a unit of text after tokenization. It can be a word, subword, or character.

   - Example: _"Hello, world!"_ → Tokens: `["Hello", ",", "world", "!"]`.

5. **Lexicon**  
   A **lexicon** is a collection of words and their meanings, often used in NLP dictionaries or linguistic databases.

   - Example: A sentiment lexicon containing words labeled as **positive** or **negative**.

6. **Vocabulary (Vocab)**  
   The **vocabulary** of an NLP model is the set of unique tokens it recognizes. It can be built from a corpus.

   - Example: If a model’s vocab is `{“hello”, “world”, “NLP”}`, it cannot understand words outside this set.

7. **Out-of-Vocabulary (OOV) Words**  
   Words not present in a model’s vocabulary. Subword tokenization techniques (like BPE, WordPiece) help handle OOV words.

   - Example: If `“blockchain”` is OOV, it may be split as `["block", "##chain"]` in WordPiece tokenization.

8. **Stopwords**  
   Common words like _"the"_, _"is"_, and _"and"_, which are often removed in NLP tasks to improve efficiency.

   - Example: Removing stopwords from _"The cat is sleeping."_ → `["cat", "sleeping"]`.

9. **Stemming**  
   Reducing words to their root form by chopping off suffixes.

   - Example: _"running"_ → _"run"_, _"jumps"_ → _"jump"_ (but may not always be a real word).

10. **Lemmatization**  
    A more accurate way to reduce words to their base form, considering grammar.

- Example: _"running"_ → _"run"_, _"better"_ → _"good"_.
