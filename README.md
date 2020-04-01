# Circle Loss

Pytorch implementation of the paper "Circle Loss: A Unified Perspective of Pair Similarity Optimization"

https://arxiv.org/abs/2002.10857



This implementation is still under developement, and I'm not sure if this code consistent with the paper.
Given class-level labels, this code calculate the similarity scores between x and weight vectors w in the paradigm of approaches like ArcFace. Not sure that Circle loss is in the same paradigm, but they don't mention K and P in the part of the face recognition (so that it seems to be).
For pair-wise labels, another implementation https://github.com/xiangli13/circle-loss is suggested.



It has been said that the official implementation will be included in https://github.com/MegEngine.



