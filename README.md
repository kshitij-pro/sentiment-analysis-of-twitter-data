# sentiment-analysis-of-twitter-data

This project's aim, is to explore the world of Natural Language Processing (NLP) by building what is known as a Sentiment Analysis Model. A sentiment analysis model is a model that analyses a given piece of text and predicts whether this piece of text expresses positive or negative sentiment

we will be using the sentiment140 dataset containing data collected from twitter. An impressive feature of this dataset is that it is perfectly balanced (i.e., the number of examples in each class is equal).

After a series of cleaning and data processing, and after visualizing our data in a word cloud, we will be building a Naive Bayezian model. This model's goal would be to properly classify positive and negative tweets in terms of sentiment. Next, we will propose a much more advanced solution using a deep learning model: LSTM. This process will require a different kind of data cleaning and processing. Also, we will discover Word Embeddings, Dropout and many other machine learning related concepts.

Throughout this notebook, we will take advantage of every result, visualization and failure in order to try and further understand the data, extract insights and information from it and learn how to improve our model. From the type of words used in positive/negative sentiment tweets, to the vocabulary diversity in each case and the day of the week in which these tweets occur, to the overfitting concept and grasping the huge importance of the data while building a given model, I really hope that you'll enjoy going through this notebook and gain not only technical skills but also analytical skills from it.

## Tokenization

In order to feed our text data to a classification model, we first need to tokenize it.
Tokenization is the process of splitting up a single string of text into a list of individual words, or tokens.

Python has a built in string method: string.split(), that splits up any given string into a list based on a splitting character (if not specified, will default to white space).

In this example, we will use the TweetTokenizer; a Twitter-aware tokenizer provided by the nltk library. In addition to a standard tokenizer, this tokenizer will split the input text based on various criterions that are well suited for the tweets use case.

## Lemmatization

According to the Cambridge English Dictionary, Lemmatization is the process of reducing the different forms of a word to one single form, for example, reducing "builds", "building", or "built" to the lemma "build". This will greatly help our classifier by treating all variants of a given word as being references to the original lemma word. For example, it will avoid interpreting "running" and "run" as completely different inputs.

In this example, we will use nltk's WordNetLemmatizer to accomplish this task. This lemmatizer however takes as input two arguments: a list of tokens to be lemmatized as well as their corresponding part of speech. The most common parts of speech in english are nouns and verbs. In order to extract each token's part of speech, we will utilize nltk's post_tag function, that takes an input a list of tokens, and returns a list of tuples, where each tuple is composed of a token and its corresponding position tag. Various position tags can be outputted from the pos_tag function, however the most notable ones are:

    NNP: Noun, proper, singular
    NN: Noun, common, singular or mass.
    VBG: Verb, gerund or present participle.
    VBN: Verb, past participle.

## Visualizing the Data

Word Clouds are one of the best visualizations for words frequencies in text documents.
Essentially, what it does is that it produces an image with frequently-appearing words in the text document, where the most frequent words are showcased with bigger font sizes, and less frequent words with smaller font sizes

## Naive Bayesian Model

Now that our data is somewhat clean, we can use it to build our classification model. One of the most commonly used classification models in Natural Language Processing (NLP) is the Naive Bayesian.
Naive Bayesian classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but rather a family of algorithms where all of them make the following naive assumptions:

    All features are independent from each other.
    Every feature contributes equally to the output.

In our case, these two assumptions can be interpreted as:

    Each word is independent from the other words, no relation between any two words of a given sentence.
    Each word contributes equally, throughout all sentences, to the decision of our model, regardless of its relative position in the sentence.

"This is bad" / "This is very bad" or "Such a kind person" / "This kind of chocolate is disgusting", in both cases the Naive Bayesian classifier would give the same importance for the words 'bad' and 'kind', albeit them having a stronger meaning and a different meaning respectively in first and second sentences.

Nevertheless, Naive Bayesian are widely used in NLP and they often output great results.

The Bayes' Theorem describes the probability of an event $A$, based on prior knowledge of conditions $B$ that might be related to the event: $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$.<br> In our case, this can be intuitively interpreted as the probability of a tweet being positive, based on prior knowledge of the words inside the input text. In a nutshell, this probability is: the probability of the first word occuring in a positive tweet, times, the probability of the second word occuring in a positive tweet, ..., times, the probability of a tweet being positive. This can be mathematically written as: $P(A \mid B) \propto P(B_1 \mid A)\times P(B_2 \mid A) \times \cdot \cdot \cdot \times P(B_n \mid A)\times P(A)$.

75.5% accuracy on the test set training a very Naive (😉) algorithm and in just 16 seconds!

Taking a look at the 20 most informative features of the model, we can notice the high volume of negative to positive (0:1) informative features. This is very interesting as it means that negative tweets have a much more concentrated and limited vocabulary when compared to positive tweets.

I personally interpret this as follows: Whenever people are in a bad mood, they are confined in such a limited space of words and creativity, in contrast with when they are in a happy mood.

## Deep Learning Model - LSTM

Deep Learning is a very rapidly growing field, that is proving to be extremely beneficial in various scenarios. One of those scenarios, which we will be studying in this notebook, is the ability to process text data in a much more complex and powerful manner. In fact, in the next section of the notebook we will be focusing on implementing a Deep Learning model that will successfully tackle and solve the above mentioned shortcomings of the Naive Bayes model, such as the lack of relationship between words in a sentence and the poor generalization on previously unseen data.

A Long Short-Term Memory, or LSTM, is a type of machine learning neural networks. More specifically, it belongs to the family of Recurrent Neural Networds (RNN) in Deep Learning, which are specifically conceived in order to process temporal data. Temporal data is defined as data that is highly influenced by the order that it is presented in. This means that data coming before or after a given datum (singular for data) can greatly affect this datum. Text data is an example of temporal data. For example, let's consider the following sentence:

    Jane is not very happy. She's still mad at you!

In the above sentence, the word not greatly influences the meaning of the upcoming words very happy. Also, we used the word she as we are speaking about a female subject.

Also, here's a fun example conveying the influence of words' positions directly influencing a sentence's meaning:

    Are you as clever as I am?

    Am I as clever as you are?

LSTM is an advanced and complex deep learning architecture, so we will avoid explaining it in detail in this notebook as it will result in a huge notebook! (Maybe it's a project for the future? 😉)

That being said, you don't really need to know the ins and outs of LSTM in order to walk through the rest of this notebook, so don't worry about it for the moment!

## Data Pre-processing

In order to feed our text data to our LSTM model, we'll have to go through several extra preprocessing steps.

Most neural networks expect numbers as inputs. Thus, we'll have to convert our text data to numerical data.

One way of doing so would be the following: collect all possible words in our dataset and generate a dictionary containing all unique words in our text corpus, then sort all of these words alphabetically and assign to each word an index. So for example, let's say our dictionary's length turned out to be 100,000 words. The word "a" would be assigned the index 0, the word "aaron" would be assigned the index 1, and so on, until we reach the last word in our dictionary, say "zulu", and assign to it the index 99,999. Great! Now each word is represented with a numerical value, and we can feed the numerical value of each word to our model.

It turns out that this step alone is not enough to be able to train good Deep Learning models. If you think about it, when the model reads an input 20,560 and then another input 20,561 for example, it would assume that these values are "close". However, those inputs could be the indexes of totally unrelated words, such as "cocktail" and "code", appearing right next to each other in the sorted dictionary. Hoping I've convinced you with this example, and that you hopefully believe that "cocktail" and "code" are, and should always be, completely unrelated, let's take a look at one solution that is widely adopted in various NLP implementations.

Also, one simple solution for this problem is to use One-Hot vectors to represent each word, but we won't bother with One-Hot vectors in this notebook, as we will be discussing a much more robust solution.
## Word Embeddings

Word embeddings are basically a way for us to convert words to representational vectors. What I mean by this is that, instead of mapping each word to an index, we want to map each word to a vector of real numbers, representing this word.

The goal here is to be able to generate similar or close representational vectors for words that have similar meaning. For example, when feeding the words "excited" and "thrilled" to the word embedding model, we would like the model to output "close" representations for both words. Whereas if we feed the words "excited" and "Java", we would like the model to output "far" representations for both words.

    The concept of "close" and "far" vectors is actually implemented using the cosine similarity. In fact, word embeddings and distance between words or relation between words is an immense discussion in its own. So I'll just keep my explanation to a minimum in this notebook.

 ## Global Vectors for Word Representation (GloVe)

Building and training good word embeddings is a tremendous process requiring millions of data samples and exceptional computational power. Luckily for us, folks at the University of Stanford already did this for us and published their results for free on their official website! Their model is called GloVe, and it's going to be what we'll use in the next steps
