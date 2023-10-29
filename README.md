<a name="br1"></a> 

**NLP Project**

**Micha Briskman – 208674713, Shlomo Gulayev - 318757382**

**Music-Genre-Classification-using-lyrics**

**Abstract**

This project aims to build a system that can identify the genre of a song based on its lyrics.

We create a set of songs with four labels - Rock, Hip-Hop, Country, Pop. Then we design four

models to classify the songs into their genres - Bert model, Glove model, Word2Vec model

and SVM model. All our models use recurrent neural networks (except SVM) to predict the

genre of the song. We implemented pytorch and used transformers text embeddings in Bert

model. The Bert model uses LSTM. For Glove model we implemented pytorch and used

Glove embeddings. W2V uses W2V embeddings. They both use LSTM. For SVM model we

implemented scikit learn algorithm and used our own word embeddings.

**Introduction**

In the field of Natural Language Processing, the classification of genres of a song solely

based on the lyrics is considered a challenging task, because audio features of a song also

provides valuable information to classify the song into its respective genre. Previously

researchers have tried several approaches to this problem, but they have failed to identify a

method which would perform significantly well in this case. SVM, KNN and Naive Bayes have

been used previously in lyrical classification research. But classification into more than 10

genres have not been particularly successful, because then the clear boundary between the

genres is often lost. So, we try to use a dataset of four genres. Hence, we try to approach

this problem as a supervised learning problem applying several methods. We analysed the

relative advantages and disadvantages of each of the methods and finally reported our

observations. With the advent of deep learning, it has been observed that Neural Networks

perform better than the previously used models.

**Dataset**

The dataset for this problem was not abundant mostly due to copyright issues. However,

after comparing datasets from several sources, we found out a data set which was most

suited for our purpose. The dataset is basically a collection of 380000+ lyrics from songs

scraped from

[https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/origin](https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/original_cleaned_lyrics.zip)

[al_cleaned_lyrics.zip](https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/original_cleaned_lyrics.zip)[ ](https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/original_cleaned_lyrics.zip). However, the dataset was so large that we didn’t have enough ram in

the notebook to process it, so we used only 97,000+ lyrics.

The structure of the data is index/song/year/artist/genre/lyrics. The data was not properly

structured according to our needs there were additional genres we didn’t need because

they were underrepresented. So, we had to process our data before it could be fitted to any

model for classification. Initially, we had to remove some irrelevant data from our dataset,

making it more compact and easier to access. Like we removed artist and song year



<a name="br2"></a> 

information thus creating just lyrics and genre mapping in our dataset. **Then we extracted**

**songs of four genres - Rock, Hip-Hop, Pop, Country.** We extracted 2557 songs from each

genre, making the dataset practical and easy to analyse. Then we removed some songs

which had very few words in its lyrics (kept more than 100 and under 400 words). Then we

tokenized the lyrics text using NLTK tool in Python for SVM model, and Spacy library for the

other models. We also did some pre-processing of data for each of our models, which would

be explained later.

**Data Analysis**

Before pre-processing we analysed the data and identified the features of data which is the

first step of any machine learning problem. This analysis helped us understand the features

of the data that would be most useful for the task in our hand. Then we calculated the

average number of unique words in each genre. This would help us understand any

correlation between the words used in lyrics and the genre type.

**Word count -** hip-hop has the most word count, and jazz has the lowest word count.

**Genre Analysis and Cleaning**

We removed some genres that were underrepresented: Folk, indie, R&B, other, metal, jazz,

electronics.

**We balanced the genres – 2557 each genre.**



<a name="br3"></a> 

**Approaches**

We have used four models: SVM model, GloVe model, Word2Vec model and Bert model.

The W2V model that we used was the most effective of all, we used a LSTM Network, with

W2V word embeddings.

**Glove Model:**

The GloVe model is an [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)[ ](https://en.wikipedia.org/wiki/Unsupervised_learning)[learning](https://en.wikipedia.org/wiki/Unsupervised_learning)[ ](https://en.wikipedia.org/wiki/Unsupervised_learning)algorithm for obtaining vector representations for

words. This is achieved by mapping words into a meaningful space where the distance between

words is related to semantic similarity. The GloVe model we used was ‘glove-wiki-gigaword-300’. We

trained Glove vectors using python's genism library. We preprocessed the data with the spacy library.

It uses a LSTM network we have built.

**Word2Vec model**

We used the word vectors (word2vec) to represent our lyrical text. These semantic vectors preserve

most of the relevant information in a text while having relatively low dimensionality. Word2Vec is an

algorithm that takes every word in your vocabulary that is, the text that needs to be classified is

turned into a unique vector that can be added, subtracted, and manipulated in other ways just like a

vector in space. The Word2Vec model we used was: ‘word2vec-google-news-300‘. We trained word

vectors using python's genism library. We preprocessed the data with the spacy library. It uses a

LSTM network we have built.

**Bert Model**

BERT is based on the [transformer](https://en.wikipedia.org/wiki/Transformer_\(machine_learning_model\))[ ](https://en.wikipedia.org/wiki/Transformer_\(machine_learning_model\))architecture. Specifically, BERT is composed of Transformer

encoder layers. We preprocessed the data with the spacy library It uses a LSTM network that we

have built.



<a name="br4"></a> 

**SVM model**

With our features and labels ready we fed them into a classier and trained it. We used 4:1

split of the dataset for training and testing. We preprocessed the data with NLTK library. We

used python's scikit learn library to implement the algorithms:

**Note:** everything is shown in the notebook

**Results**

Now we report the results of experiments on these models on a dataset of 10,228

(2557\*4) songs equally distributed among all the genres.

We ran the models (except SVM) with the following parameters:

•

•

•

learning rate 10<sub>-5</sub>

dropout probability: 0.2

The number of epochs: Providing enough time for the model to learn about

25 epochs.

•

•

•

batch size: Tried batch sizes of 96.

LSTM layers: 2

hidden dim: 128

In the Glove model we could achieve an accuracy of 58%.

In the W2V model we could achieve an accuracy of 64%.

In the Bert model we could achieve an accuracy of only 46%.

In the SVM model we could achieve an accuracy of only 29%.



<a name="br5"></a> 

Accuracy percent on train set

**64%**

**58%**

**46%**

**29%**

W2V

Glove

Bert

SVM

**Loss function and accuracy Glove model:**

**Evaluation Glove on test set:**



<a name="br6"></a> 

**Loss function and accuracy Word2Vec model:**

**Evaluation Word2Vec on test set:**



<a name="br7"></a> 

**Loss function Bert:**

**Evaluation Bert on test set:**

**Evaluation SVM:**



<a name="br8"></a> 

**Conclusions and Future Work**

From the models that we developed and the experiments that we conducted we can

say that the Word2Vec Model performed well compared to the other models. In that

respect Glove model perform well as well. Apart from Rock (as seen from the

confusion matrixes) other genres might be mislabelled at times. However, limited by

time/RAM/GPU, we could produce some significant results in the field of music genre

classification based on lyrics. There is a lot that can be done like better pre-

processing of data. Adding more data for each of genre classes. We might train a

model with lyrics as well as audio features and it is expected that we can get better

results. Also, we might train a more complex model which would remember order of

words, and we can experiment on our training data. Classification by lyrics will always

be inherently awed by vague genre boundaries with many genres borrowing lyrics

and styles from one another. For example, one merely need consider cover songs

which utilise the same lyrics but produce songs in vastly different genres, songs

which have no lyrical content. To produce a state of the art classifier is evident that

this classifier must take into account more than just the lyrical content of the song.

Audio data typically performs the strongest and further research could look into

employing these models to the audio and symbolic data and combining with the

lyrics to build a stronger classifier.

