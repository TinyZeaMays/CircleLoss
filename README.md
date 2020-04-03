# Circle Loss

Pytorch implementation of the paper "Circle Loss: A Unified Perspective of Pair Similarity Optimization"

https://arxiv.org/abs/2002.10857



This implementation is still under developement, and I'm not sure if this code consistent with the paper.



# Update

To avoid overflow, CircleLossBackward directly backward with gradients. The loss returned has already been detached. I just changed a sign in Eq 10.



CircleLossLikeCE is my early implementation which was named with CircleLoss and is replaced by CircleLossBackward.

Given class-level labels, CircleLossLikeCE calculate the similarity scores between x and weight vectors w in the paradigm of approaches like ArcFace. 



For pair-wise labels, another implementation https://github.com/xiangli13/circle-loss is suggested.



# Other

It has been said that the official implementation will be included in https://github.com/MegEngine.



Is there something wrong in Eq. 10?



