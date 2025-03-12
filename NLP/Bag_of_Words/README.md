# Bag Of Words

#### What is Bag-of-Words?

![](https://miro.medium.com/max/300/0*JpqZhCNsQ_OGaRkB.jpg)

We need a way to represent text data for machine learning algorithm and the bag-of-words model helps us to achieve that task. The bag-of-words model is simple to understand and implement. It is a way of `extracting features` from the text for use in machine learning algorithms.

In this approach, we use the tokenized words for each observation and find out the frequency of each token.

Now, **What is Tokenization?**

Before tokenization we need to konw some important terms:

- corpus : Collection of text documents
- corups > Documents > Paragraphs > Sentences > Tokens
- Tokens : Smaller units of a text (words, phrases, ngrams)
- Ngrams : group of n words togather

![](https://github.com/patelvini/ML/raw/0ba4639e4adb2b746aca0397c63deb21754193ef/NLP/ngrams.PNG)

**Tokenization**

- Process of splitting a text object into smaller units (tokens).
- smaller units : words, numbers, symbols, ngrams, characters

Two types of tokenization :
1. white space tokenizer / unigram tokenizer

![](https://github.com/patelvini/ML/raw/0ba4639e4adb2b746aca0397c63deb21754193ef/NLP/white_space_tokenizer.PNG)

2. Regular expression tokenizer

![](https://github.com/patelvini/ML/raw/0ba4639e4adb2b746aca0397c63deb21754193ef/NLP/Regx_tokenizer.PNG)

Let’s take an example to understand this concept in depth.

#### Step 1: Collect Data
```
“It was the best of times”
“It was the worst of times”
“It was the age of wisdom”
“It was the age of foolishness”
```
We treat each sentence as a separate document and we make a list of all words from all the four documents excluding the punctuation. 

#### Step 2: Design the Vocabulary

We get,
```
‘It’, ‘was’, ‘the’, ‘best’, ‘of’, ‘times’, ‘worst’, ‘age’, ‘wisdom’, ‘foolishness’
```
#### Step 3: Create Document Vectors

The next step is to score the words in each document.

The objective is to turn each document of free text into a vector that we can use as input or output for a machine learning model.

Because we know the vocabulary has 10 words, we can use a fixed-length document representation of 10, with one position in the vector to score each word.

The simplest scoring method is to mark the presence of words as a boolean value, 0 for absent, 1 for present.

Using the arbitrary ordering of words listed above in our vocabulary, we can step through the first document (“It was the best of times“) and convert it into a binary vector.

The scoring of the document would look as follows:
```
“it” = 1
“was” = 1
“the” = 1
“best” = 1
“of” = 1
“times” = 1
“worst” = 0
“age” = 0
“wisdom” = 0
“foolishness” = 0
```
Rest of the documents will be:
```
“It was the best of times” = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
“It was the worst of times” = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
“It was the age of wisdom” = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
“It was the age of foolishness” = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
```

All ordering of the words is nominally discarded and we have a consistent way of extracting features from any document in our corpus, ready for use in modeling.

New documents that overlap with the vocabulary of known words, but may contain words outside of the vocabulary, can still be encoded, where only the occurrence of known words are scored and unknown words are ignored.

### Managing Vocabulary

As the vocabulary size increases, so does the vector representation of documents.

In the previous example, the length of the document vector is equal to the number of known words.

You can imagine that for a very large corpus, such as thousands of books, that the length of the vector might be thousands or millions of positions. Further, each document may contain very few of the known words in the vocabulary.

This results in a vector with lots of zero scores, called a sparse vector or sparse representation.

Sparse vectors require more memory and computational resources when modeling and the vast number of positions or dimensions can make the modeling process very challenging for traditional algorithms.

As such, there is pressure to decrease the size of the vocabulary when using a bag-of-words model.

There are simple text cleaning techniques that can be used as a first step, such as:

- Ignoring case
- Ignoring punctuation
- Ignoring frequent words that don’t - contain much information, called stop words, like “a,” “of,” etc.
- Fixing misspelled words.
- Reducing words to their stem (e.g. “play” from “playing”) using stemming algorithms.

A more sophisticated approach is to create a vocabulary of grouped words. This both changes the scope of the vocabulary and allows the bag-of-words to capture a little bit more meaning from the document.

In this approach, each word or token is called a “gram”. Creating a vocabulary of two-word pairs is, in turn, called a bigram model. Again, only the bigrams that appear in the corpus are modeled, not all possible bigrams.

> _An N-gram is an N-token sequence of words: a 2-gram (more commonly called a bigram) is a two-word sequence of words like “please turn”, “turn your”, or “your homework”, and a 3-gram (more commonly called a trigram) is a three-word sequence of words like “please turn your”, or “turn your homework”._

For example, the bigrams in the first line of text in the previous section: “It was the best of times” are as follows:
```
“it was”
“was the”
“the best”
“best of”
“of times”
```

A vocabulary then tracks triplets of words is called a trigram model and the general approach is called the n-gram model, where n refers to the number of grouped words.

Often a simple bigram approach is better than a 1-gram bag-of-words model for tasks like documentation classification.

> _a bag-of-bigrams representation is much more powerful than bag-of-words, and in many cases proves very hard to beat._

The process of converting NLP text into numbers is called **vectorization** in ML. 

Different ways to convert text into vectors are:
- Counting the number of times each word appears in a document.
- Calculating the frequency that each word appears in a document out of all the words in the document.

### CountVectorizer

CountVectorizer works on Terms Frequency, i.e. counting the occurrences of tokens and building a sparse matrix of documents x tokens.

### TF-IDF Vectorizer

TF-IDF stands for term frequency-inverse document frequency. TF-IDF weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

- **Term Frequency (TF):** is a scoring of the frequency of the word in the current document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. The term frequency is often divided by the document length to normalize.

  ![](https://miro.medium.com/max/404/1*SUAeubfQGK_w0XZWQW6V1Q.png)

- **Inverse Document Frequency (IDF):** is a scoring of how rare the word is across documents. IDF is a measure of how rare a term is. Rarer the term, more is the IDF score.

  ![](https://miro.medium.com/max/411/1*T57j-UDzXizqG40FUfmkLw.png)

Thus,

  ![](https://miro.medium.com/max/215/1*YrgmAeG7KNRB4dQcGcsdyg.png)

### Word Hashing
You may remember from computer science that a hash function is a bit of math that maps data to a fixed size set of numbers.

For example, we use them in hash tables when programming where perhaps names are converted to numbers for fast lookup.

We can use a hash representation of known words in our vocabulary. This addresses the problem of having a very large vocabulary for a large text corpus because we can choose the size of the hash space, which is in turn the size of the vector representation of the document.

Words are hashed deterministically to the same integer index in the target hash space. A binary score or count can then be used to score the word.

This is called the “hash trick” or “feature hashing“.

The challenge is to choose a hash space to accommodate the chosen vocabulary size to minimize the probability of collisions and trade-off sparsity.

## Limitations of Bag-of-Words
The bag-of-words model is very simple to understand and implement and offers a lot of flexibility for customization on your specific text data.

It has been used with great success on prediction problems like language modeling and documentation classification.

Nevertheless, it suffers from some shortcomings, such as:

- **Vocabulary:** The vocabulary requires careful design, most specifically in order to manage the size, which impacts the sparsity of the document representations.
- **Sparsity:** Sparse representations are harder to model both for computational reasons (space and time complexity) and also for information reasons, where the challenge is for the models to harness so little information in such a large representational space.
- **Meaning:** Discarding word order ignores the context, and in turn meaning of words in the document (semantics). Context and meaning can offer a lot to the model, that if modeled could tell the difference between the same words differently arranged (“this is interesting” vs “is this interesting”), synonyms (“old bike” vs “used bike”), and much more.


