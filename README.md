# Generative Models

Summary: This project lets you dive into a new family of models that is used for generation purposes.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Discriminative and Generative models](#discriminative-and-generative-models) \
    2.2. [Variational AutoEncoder](#variational-autoencoder) \
    2.3. [Generative Adversarial Network](#generative-adversarial-network)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

In this project, we will get acquainted with another machine learning problem that we have not identified before. With the task of building generative models - the goal of which is to create data that looks real. And in fact, we have even trained such models before: let's recall, for example, the name generator from the project with recurrent networks. Another, but not so obvious example, is the PCA model, which we trained to compress the feature space. But not the direct use of this model, but inverse - when we restore the original vector from the latent representation.

So it comes out that our task is to, having a set of observations, in some sense understand the process that generates them and then reproduce it. The importance of this task is due to the fact that our world stores a lot of various information, most of which a person can process simply. But how to teach a machine to analyze and understand the nature of this information? As soon as we find the answer to this question, it will become easier for us to realize the potential from these models.

If you follow the news related to AI, you have probably already heard about well-known implementations of generative models: such as ChatGPT and LaMDA for text generation, Dall-e and MidJourney for generating images from the description, Photoshop++ for image modifications on demand, DLSS for generating images with upscaled resolution, many different models for text voicing, MuseNet for generating music, models for generating videos and much more. And each of these models immediately found a pool of applications where it can be useful for humans. In this project, we will get acquainted with the main architectures that are used to solve this problem.

## Chapter II. Introduction

### Discriminative and Generative models

In parallel with the classification of machine learning tasks into supervised and unsupervised, there is an alternative division into Discriminative and Generative. In simple words, the difference is as follows:

- Generative models generate new data instances. In a probabilistic setting, the model tries to learn the joint distribution of data p(x, y) (or p(x) if there is no target variable) in a multidimensional space
- Discriminative models distinguish between different kinds of data instances. That is, the model tries to learn the conditional probability p(y|x)

It can be seen from the statements that generative models, in fact, solve a more complex problem than similar discriminatory models. While a discriminatory model needs to find a pattern by which it can distinguish dogs from cats, generative models need to match all parts of the animal's body, how they should interact with the environment, and much more. Generative models should model more, many times more.

This is also confirmed mathematically - the search for a probability distribution in a multidimensional space is quite a difficult task, so the task is often simplified. One of the main approaches is to introduce a latent variable z and try to learn a function that transforms z into an observable variable x. This trick is actually very familiar to man. If we encounter complex patterns, we try to describe them with a small number of variables. For example, instead of describing the motion of a car as a set of positions of atoms, we describe its position in terms of initial coordinates, set the velocity and acceleration, and simply apply the equation of motion. It is only worth mentioning that z in this case is usually given as a random multidimensional vector using simple distributions. Thus, $`P(x) = \int P(x|z)P(z)dz`$

As a result our task is to train the model for $`P(x|z)`$, for which neural networks suit very well.

### Variational AutoEncoder

AutoEncoder is an architecture of the neural network that compresses input information and decompresses it in such way that it should restore the input vector:

![AutoEncoder](misc/images/autoencoder.png)

Actually on this step we get what we want: compressed representation is z and decoder is a function to restore the real vector. And this will work great! All we need is to understand how compressed representations are distributed in low dimensional space.
But we also want to introduce you to how neural networks could be easily used to optimize distributions. To do this we:

1. replace our compressed representation layer with random vector that distributed normally ~N(mean, std) where mean and std defines with neural network
2. randomly sample our compressed representation from normal distribution and pass it to decoder:

![Encoder](misc/images/encoder.png)

The benefit of such modification is that the model now could use uncertainty during pattern fitting. If the model has not seen a lot of examples of people with glasses, the representations of these instances will have high deviation. Besides, it helps the model to avoid overfitting and have better smooth latent space since every input can be sampled into similar but different representations. For any sampling of the latent distributions, we're expecting our decoder model to be able to accurately reconstruct the input:

![reconstruct](misc/images/reconstruct.png)

Together with the modification of architecture it’s better to modify loss function for neural network training. For VAEs we use ELBO. To derive what it is and why we use it is part of your task.

Please also note that both of these models could be used for dimensionality reduction tasks.

### Generative Adversarial Network

To finally convince you that neural networks are a very flexible tool, we will consider the GAN model. Imagine that you want to teach a model to play chess, but you do it without any stored games from other players, but by making the computer play online. And at the same time, you are making the computer play against another model. Who will win then? Of course, if the models are “raw”, it can hardly be called the game. But if the loss functions are set up correctly, then the models will begin to understand the game while playing, and subsequently they will be able to beat even the most experienced grandmasters. Generative **Adversarial** Networks do something similar. Within such a network, there are 2 models that play against each other. The first player is the generator trying to learn how to imitate real data (vectors x) and the second player is the discriminator trying to learn how to distinguish the real world data from the one created by the generator. In other sources this task is often described using an example, where the generator is a scammer who prints banknotes in order to put them in a bank - discriminator.

Let's take a closer look at the architecture. Just like in VAE, we represent our data using a random hidden (latent) variable, which we feed into the generator. The generator imitates real data and gives it to the discriminator. In parallel, real data comes to the input of the discriminator, which the discriminator processes simultaneously with artificial data. Since the 2 models have different goals, they cannot train at the same time. Therefore, GAN has 2 loss functions, each of which updates its part of the network. As a result every time the discriminator notices a difference between the two distributions the generator adjusts its parameters slightly to make it go away, until at the end.

![Generative Adversarial Network](misc/images/generative_adversarial_network.png)

Describing how the loss function works for each model within a GAN is part of your assignment.

## Chapter III. Goal

The goals of this project are to understand the architecture of the main generative models: VAE and GAN

## Chapter IV. Instructions

- This project will only be evaluated by humans. You are free to organize and name your files as you desire.
- Here and further we use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try Google Colab. It offers kernels (Runtime) with GPU for free that is faster than CPU for such tasks. The norm is not applied to this project. Nevertheless, you are asked to be clear and structured in the conception of your source code.
- Store the datasets in the subfolder data

## Chapter V. Task

### Dataset

Mnist - [source](http://yann.lecun.com/exdb/mnist/)

“Hello world” dataset for hand-written digit recognition. Many libraries have built-in tools to load this dataset.

![Dataset](misc/images/dataset.png)

### Task

1. Answer questions from Preamble and Introduction \
    a. Derive the formula for ELBO and explain why this loss is used for VAE training \
    b. Explain which loss is used for generator and discriminator inside GAN
2. Generative model from Variational AutoEncoder \
    a. Train VAE using MNIST dataset. Use a 2-dimensional vector as a latent vector. Use ELBO as a loss function \
    b. Plot distribution of latent vectors using different colors for every digit \
    c. Create 15x15 mesh in 2-dimensional latent space. Use decoder as a generator model to reconstruct hand-written digits. Combine all reconstructed digits in one 15x15 figure. \
    d. What can you note from the plots?
3. Supervised generative model from VAE \
    a. Train Supervised Variational AutoEncoder using MNIST dataset where: \
        i. encoder return 1-dimensional latent vector \
        ii. decoder’s input consists from latent vector and digit label (as one  hot vector) \
    b. Create mesh from 15 points in latent space. Combine them with 10 digit labels and plot 15x10 reconstructed digits. What can you note from the plots?
4. Generative Adversarial Network \
    a. Implement Generator that uses 2-dimensional random vector as input \
    b. Implement Discriminator to classify real and fake images \
    c. Train GAN to create hand-written digits \
    d. Plot loss progress both for generator and discriminator? Who wins? \
    e. Create 15x15 mesh in 2-dimensional latent space. Use Generator to reconstruct hand-written digits. Combine all reconstructed digits in one 15x15 figure. \
    f. What can you note from the plot?
5. Gradient Reversal \
    a. Explain what this layer is and why is it useful? \
    b. Instead of using 2 separate backward steps for generator and discriminator use 1 but with gradient reversal layer \
    c. Achieve similar quality
6. Supervised Generative Adversarial Network \
    a. Think about how you can modify GAN to make it possible to control what number should be generated. Explain what you will modify \
    b. Implement the modification and train the model \
    c. Visualize all 10 generated hand-written digits with the same random part of the input variable.

### Submission

Your repository should contain one or several notebooks with your solutions.

## Chapter VI. Bonus part

- What is the idea of the Diffusion Model? Try to implement it on the mnist dataset

💡 [Tap here](https://forms.yandex.ru/cloud/646b47de73cee708d97f0b1a/) **to leave your feedback on the project**. Pedago Team really tries to make your educational experience better.
