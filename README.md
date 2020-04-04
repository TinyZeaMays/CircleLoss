# Circle Loss

An unofficial Pytorch implementation of the paper "Circle Loss: A Unified Perspective of Pair Similarity Optimization".

https://arxiv.org/abs/2002.10857



# Update

Use CircleLoss in circle_loss.py, and there is a simple example of mnist in mnist_example.py.



For pair-wise labels, another implementation https://github.com/xiangli13/circle-loss is suggested.



# Early

Sorry for using master branch as dev. Some early implementations are kept in  circle_loss_early.py. 



CircleLossLikeCE is an early implementation to use CircleLoss in the paradigm of approaches like ArcFace. It only consists with the paper on a special case.



CircleLossBackward is an early implementation to avoid overflow in the method of applying backward with handcraft gradients. A negative sign is added to the Eq. 10 in this code to fix the equation. It's correct, stable but messy.



# Other

It has been said that the official implementation will be included in https://github.com/MegEngine.



Thanks very much for Yifan Sun's advice!

