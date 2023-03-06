# Masked Face recognition using Barlow Twins technique

We proposed a new solution for masked face recognition. [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf) technique present a new loss which makes the networks capable or learing invariance to particular transformations. We tought to apply this technqieu considering the presence of the face mask as particular transformation so that the network learns the invariance to its presence.  
Our system is composed of a neural network as a feature extractor plus a k-NN as final classifier.  
For training the system, we used the [MLFW dataset](https://arxiv.org/pdf/2109.05804.pdf).  
We tested the system only on identities not present in the training set and it was able to reach an accuracy level of 78.89.  
