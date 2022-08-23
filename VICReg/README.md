# VICReg

This is a tensorflow implementation of the architecture: VICReg (Meta AI). I tested the architecture for a multi-modal setting using an image-captions dataset. However, the code can easily be adapted to support any kind of encoder (e.g time series, text, audio...). See the the VICReg article in the [wiki](https://github.com/gmbss0/papers-implemented/wiki/VICReg) to find more details about this paper.

## Learnings
I wanted to test VICReg on multi-modal data, as this seems to be the main selling point of this joint-embedding architecture. Moreover, the authors claim that it is much easier to get to work VICReg, compared to contrastive approaches or self-distillation-based approaches. The dimension of the embeddings and representations is challenging for many GPU set-ups... I tried to replicate the results with smaller expander dimension and smaller embedding, but did not succeed. I still believe that this is a really cool work and the simplicity of the loss terms still fascinates me. Another point I would like to experiment with in the future is the sensitivity of VICReg to different data augmentations, as this still seems to be crucial for everything to work.
