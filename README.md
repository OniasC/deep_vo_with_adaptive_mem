# deep_vo_with_adaptive_mem
My implementation of Deep Visual Odometry with Adaptive Memory

It is based on this paper (link here).

The first part of the model is the encoding section of flownet. The model was taken from (link here) and the pretrained values from (link here). Then, the tracking part is composed of a convLSTM (model taken from link here, inspired by this paper - link here). There are two other sections that i'll discuss later.


### Datasets

I'm interested in reproducing results from TUM dataset, which was split in training/testing arbitrarily by the paper's authors.

Im using the same split they used. For mocking, im using only 3 datasets from TUM:

- training:
    - fr1/desk
    - fr2/xyz

- test:
    - fr2/desk