# CS 680 Final Project

This is my final project for CS 680, Introduction to Machine Learning, at the University of Waterloo. Here we attempt to generate synthetic handwriting from a sequence of images. We explore generation with two methods, an encoder-decoder, and a generative adversarial network.




An informal post mortem statement (more of a rant than anything substantial):

Although neither of these methods worked in the way that I had originally planned, this was a great learning experience both for exposure into machine learning research as well as on a personal level.

The most important lesson (that I learned the hard way) was to check my evaluations regularly. At the beginning of the project I believe that I proposed a project that was inconcievable given the timeframe, my current workload, my current skill level, and my knowledge of the packages that I had planned on using. I over evaluated how much time I would have to put into this project, as well as the tools that I would be using. At every step of this project, I felt that whenever I made some sort of progress, I would require twice the time and effort to make it too the next stage. I understood that being ambitious was an asset, however my initial enthusiasm was far outshadowed by my lack of self assessment.

When first implementing the GAN models with Keras, I had thought that more customization to the reccurent layers was possible. Similarly, I had thought that after implementing a generator, which did indeed generate the a sequence of pen points as desired, and implementing a discriminator, which did in fact return a set of probabilities based on the data given, that training the two models together would be easier given the high level architecture I was using. Turns out that even though I had concrete examples GAN training on other models, it was difficult to incorporate the code to be usable in my case.

As time ran low and the training still not implemented, I turned to other methods to try to produce some sort of results, and while the alternative model did train, there was limited time to experiment with model sizes, and solve the exploding loss function problem that plagued larger encoder-decoder models.
