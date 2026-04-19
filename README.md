# Spam-Ham-messages-Classifier-based-on-classical-Bernoulli-Naive-Bayes.

This is an AI model that can tell if the text message is spam or not.  
How it works:  
It determines whether a message is spam based on the words it contains.  
First, it's trained on a dataset of messages labeled as “spam” or “ham”. From this data, it learns:  
1.how often spam occurs overall => $\phi_{y}$  
2.a vector of probabilities for each word, where each value represents how likely that word appears in spam messages => $\phi_{j|1}$  
3.a similar set of probabilities for non-spam messages => $\phi_{j|0}$  
This means the model stores values like “probability that the word ‘free’ appears in spam” or “probability that ‘meeting’ appears in non-spam.” These probabilities are often written as $\phi_{j|y}$, meaning the probability of word j given a class y (spam or not spam).

When a new message comes in, the classifier looks at all the words in it and uses these learned probabilities to estimate how likely the message is to be spam. It combines the overall probability of spam and the probabilities of each word appearing in spam.  

It makes a simplifying assumption that all words are independent of each other (this is why it’s called “naive”).

In practice, it often uses logarithms instead of multiplying probabilities directly, to avoid numerical issues.

The classifier uses a learned table (or vector) of word probabilities to measure how strongly the words in a message indicate spam, and picks the more likely category.  
  
Formulas that are used in classifer:  
$P(y = 1 \mid x) = \frac{1}{1 + e^{B - A}}$  

  
$$
A = \log(\phi_y) + \sum_{j=1}^{n} \left[ x_j \log(\phi_{j|1}) + (1 - x_j)\log(1 - \phi_{j|1}) \right]
$$  
$$
B = \log(1 - \phi_y) + \sum_{j=1}^{n} \left[ x_j \log(\phi_{j|0}) + (1 - x_j)\log(1 - \phi_{j|0}) \right]
$$  
In NB for spam classification we use a vector phi$phi_{j|0}$ is a vector 
$$
\phi_y = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}(y^{(i)} = 1)
$$  
$$
\phi_{j \mid y=1} =
\frac{\sum_{i=1}^{m} \mathbb{1}\(y^{(i)} = 1, x_j^{(i)} = 1\) + 1}
{\sum_{i=1}^{m} \mathbb{1}\(y^{(i)} = 1\) + 2}
$$  
$$
\phi_{j \mid y=0} =
\frac{\sum_{i=1}^{m} \mathbb{1}\(y^{(i)} = 0, x_j^{(i)} = 1\) + 1}
{\sum_{i=1}^{m} \mathbb{1}\(y^{(i)} = 0\) + 2}
$$

$P(y = 1 \mid x)$ is a probability of message being spam.  
$m$ - number of messages.  
$n$ - number of words in vector of probabilities.  
$y=1$ - spam  
$x_j^{(i)}$ - word is in message 
