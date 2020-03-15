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
```
“It was the best of times”
“It was the worst of times”
“It was the age of wisdom”
“It was the age of foolishness”
```

We treat each sentence as a separate document and we make a list of all words from all the four documents excluding the punctuation. 

We get,
```
‘It’, ‘was’, ‘the’, ‘best’, ‘of’, ‘times’, ‘worst’, ‘age’, ‘wisdom’, ‘foolishness’
```

The next step is the create vectors. Vectors convert text that can be used by the machine learning algorithm.
We take the first document — “It was the best of times” and we check the frequency of words from the 10 unique words.
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


