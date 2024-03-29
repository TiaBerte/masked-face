# Masked Face recognition using Barlow Twins technique

We proposed a new solution for masked face recognition. [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf) technique present a new loss which makes the networks capable or learing invariance to particular transformations. We tought to apply this technqieu considering the presence of the face mask as particular transformation so that the network learns the invariance to its presence.  
Our system is composed of a neural network as a feature extractor plus a k-NN as final classifier.  
For training the system, we used the [MLFW dataset](https://arxiv.org/pdf/2109.05804.pdf).  
We tested the system only on identities not present in the training set and it was able to reach an accuracy level of 79.72%.  
A complete description of the whole project can be found in the [report](https://github.com/TiaBerte/masked-face/blob/main/report.pdf).  
Below you can see samples of the sythetic images present in the MLFW dataset.

<p align="center">
<img src="https://user-images.githubusercontent.com/33131887/223077611-aa2f4a2c-0ad0-45bb-8622-7f43e0afd041.jpg" width="150" height="150"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://user-images.githubusercontent.com/33131887/223077161-3566b698-688c-47cd-94c9-0c92ba26e72e.jpg" width="150" height="150"/>
</p>
