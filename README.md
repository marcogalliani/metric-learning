# Deep metric learning

## Resources
- [metric-learn](http://contrib.scikit-learn.org/metric-learn/user_guide.html): a python package to use traditional metric learning algorithms
- [medium post](https://medium.com/data-science/a-friendly-introduction-to-siamese-networks-85ab17522942) presenting a tutorial on siamese networks
- [Github repo](https://github.com/adambielski/siamese-triplet) containing a more advanced implementation of siamese nets with examples
- [Qdrant github repo](https://github.com/qdrant/awesome-metric-learning) with a collection of nice resources about metric learning
  
## To do
- fine tuning on tilda-data, test on leather data
- Understand better ArcFace loss
- better modularity class CNN
- load on gitgi



## Logbook
- While training using triplet loss with hard batch mining strategy, the loss per epoch has to be computed by weighting for the number of triplets extracted for the batch
- Using Data Augmentation allows to extract more triplets if using triplet loss. Moreover it provides invariance with respect to the transformations used to augment the data. Another intersting aspect that motivates the usage of augmntations in our context is the fact that after the network is trained, the training data are used as templates to classify test image using cosine similarity. Data augmentation helps the network not to memorize the training samples, but to actually learn useful features.
- using Arcface loss drastically improves performance with repect to triplet loss
- a parser for each dataset, then the loader is passed to the CustomDataset when data are added
- Textile defects datasets: Tilda dataset

## Docs
Various loss functions are available to model the metric learned by the network
- Contrastive loss: [Hadsell,Chopra,LeCun,2006](https://ieeexplore.ieee.org/document/1640964)
    $$
    \mathcal{L}(e_{1},e_{2}) = (1-L)d(e_{1},e_{2})^{2} + L[max(m-d(e_{1},e_{2}),0)]^{2}
    $$
- Triplet loss: $I$: sample image (anchor), $P$: image of the same class of $I$, $N$: image of different class of $I$
    $$
    \mathcal{L}(I,P,N) = \max\{0, m +(\|f_{w}(I)-f_{w}(N)\|-\|f_{w}(I)-f_{w}(P)\|)\}
    $$
    In other words: if we minimize the previously defined loss, the distance between $I$ and $P$ in the embedded space should be lower than the one of $I$ and $N$
- ArcFace loss




