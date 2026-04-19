# Spam-Ham-messages-Classifier-based-on-classical-Bernoulli-Naive-Bayes.

This is an AI model that can tell if the text message is spam or not.  
It uses classical Bernoulli Naive Bayes that has following formulas:  
$P(y = 1 \mid x) = \frac{1}{1 + e^{B - A}}$

  
$$
A = \log(\phi_y) + \sum_{j=1}^{n} \left[ x_j \log(\phi_{j|1}) + (1 - x_j)\log(1 - \phi_{j|1}) \right]
$$  
$$
B = \log(1 - \phi_y) + \sum_{j=1}^{n} \left[ x_j \log(\phi_{j|0}) + (1 - x_j)\log(1 - \phi_{j|0}) \right]
$$  
where  
$$
\phi_y = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 1}
$$

$P(y = 1 \mid x)$ is a probability of message being spam.
