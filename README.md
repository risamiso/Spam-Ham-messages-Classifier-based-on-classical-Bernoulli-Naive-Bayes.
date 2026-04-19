# Spam-Ham-messages-Classifier-based-on-classical-Bernoulli-Naive-Bayes.

This is an AI model that can tell if the text message is spam or not.  
It uses classical Bernoulli Naive Bayes. It uses following formulas:
$P(y = 1 \mid x) = \frac{1}{1 + e^{B - A}}$  
$P(y = 1 \mid x)$ is a probability of message being spam.




$$
A = \log(\phi_y) + \sum_{j=1}^{n} \left[ x_j \log(\phi_{j|1}) + (1 - x_j)\log(1 - \phi_{j|1}) \right]
$$
