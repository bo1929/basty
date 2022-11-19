# Overview of **basty**

The behavior mapping task consists of two essential steps:

  1. computing meaningful representations capturing the interesting characteristics and dynamics of the,
  2. analyzing and interpreting the behavioral representations to explore and categorize animal behaviors.

A recent trend for achieving the first step is applying manifold-based dimensionality reduction algorithms to high-dimensional time series of features.
Many methods, such as MotionMapper and B-SOID, embeds different behavior experiments and many animals in a single joint behavioral space.
However, we observed that animal behavior is a highly-unstructured phenomenon with a great variance among experiments.
Hence, relying on manifold-based dimensionality reduction algorithms such as UMAP and t-SNE too much and expecting them to create a joint behavioral space describing all the dynamics could easily end up in having hard-to-interpret and mixed-up behavioral representations.

Our preliminary analysis revealed that some behavioral repertoires exhibited during different experiments tend might or might not have similarities.
It is often the case that behavior is exhibited only in a subgroup of experiments.
Using this observation, we approach the problem based on the idea of providing different "views" on the behavioral repertoire that we want to map, guided by annotated experiments and predefined behavioral categories.
More explicitly, we generate a semi-supervised behavioral embedding for each annotation experiment in our data using an extension of UMAP.
Each embedding can be considered a view of the experiment we want to analyze, guided by the behavioral repertoire of the corresponding annotated experiment.
This approach is called semi-supervised pair embeddings.

So, what is the benefit of semi-supervised pair embeddings over straightforwardly generating a joint behavioral space?
Especially when the behavioral repertoires of the pair of experiments are similar, the provided “view” turns out to be an accurate, easy-to-interpret low-dimensional representation of the behavioral repertoire of the unannotated experiment.
When the behavioral repertoire and/or feature distribution are dissimilar[^1] the resulting embedding may not provide useful information about the unannotated experiment, but an advantage of this approach is that the other pair embeddings do not get distorted by poor matches, providing robustness to noise and behavioral variations.

[^1]: Worse than that, it might be the case where tracking of the annotated experiment is erroneous.

Then, the question becomes the following.
How one can combine and interpret these different views, and end up with behavior mapping?
For instance, if one has $N$ different annotated experiments, the semi-supervised embeddings approach would generate $N$ different embeddings.
To this end, we designed a nearest neighbor based analysis scheme to assign behavioral similarity scores to each time point.
Here, the main idea is computing a behavior mapping by incorporating uncertainties of the views and dissimilarities between repertoires.
So, using the categorical distributions of nearest neighbors' annotations and local distances in the semi-supervised pair embedding space, our method tune the contributions of each annotated experiment.
Moreover, our method considers the imbalanced distribution of behavior occurrences and scarcity of particular behaviors.
Computational experiments show that **basty** attains capturing rarely exhibited behaviors and even is able to detect and discover unseen unannotated behavioral categories using behavioral scores.
