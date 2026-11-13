# Build a Small GPT-Style Language Model from Scratch

A practical, self-paced learning guide to the foundations of machine
learning, neural networks, language modeling, and decoder-only
Transformers.

The goal of this guide is to help you understand the ideas behind
GPT-style language models and implement a small text-only model
yourself. It is designed to be followed at your own pace: work through
it intensively, spread it over a week, or take longer when a concept
needs more attention.

> **Project outcome:** Build and train a small decoder-only Transformer
> in Python using PyTorch, starting with randomly initialized weights
> and a text dataset. The model should learn to predict the next token
> and generate text from a prompt.

This is an educational project, not an attempt to reproduce ChatGPT. A
small model trained on a small corpus will learn limited patterns from
that corpus; it will not become a general-purpose assistant.

------------------------------------------------------------------------

## Contents

-   [1. Scope and learning principles](#1-scope-and-learning-principles)
-   [2. Prerequisites and tools](#2-prerequisites-and-tools)
-   [3. Learning roadmap](#3-learning-roadmap)
-   [4. Build the model](#4-build-the-model)
-   [5. Recommended resources](#5-recommended-resources)
-   [6. Suggested first experiment](#6-suggested-first-experiment)
-   [7. What to defer](#7-what-to-defer)
-   [8. Completion checklist](#8-completion-checklist)
-   [9. How to use AI as a tutor without outsourcing the
    work](#9-how-to-use-ai-as-a-tutor-without-outsourcing-the-work)

------------------------------------------------------------------------

## 1. Scope and learning principles

### What you will learn

By the end of the guide, you should be able to explain and implement the
main parts of a small GPT-style language model:

-   The basic machine-learning training process.
-   Vectors, matrices, tensor operations, and the derivatives needed for
    gradient-based learning.
-   Neural networks, forward propagation, loss, backpropagation, and
    optimization.
-   Language modeling and next-token prediction.
-   Tokenization, vocabulary construction, and embeddings.
-   Self-attention, causal masking, and multi-head attention.
-   The decoder-only Transformer architecture.
-   Model training, autoregressive text generation, and basic
    evaluation.

### What "from scratch" means here

For this project, "from scratch" means:

-   Start with randomly initialized model weights.
-   Write the model architecture and training loop yourself.
-   Implement the tokenization approach used in the first experiment.
-   Train the model on your chosen text corpus.
-   Implement text generation yourself.

Use PyTorch for tensors, automatic differentiation, and GPU support. You
do **not** need to implement matrix multiplication or an automatic
differentiation engine from scratch. Reimplementing those would be a
separate learning project.

Do not use a pretrained language model or a high-level GPT
implementation for the main project.

### A useful learning rule

For every major component, be able to answer:

1.  What problem does it solve?
2.  What are its inputs and outputs?
3.  What happens to the tensor shapes?
4.  Why is it needed in a language model?

Do not move on just because you have watched a video. Move on when you
can explain the concept and reproduce a small example yourself.

------------------------------------------------------------------------

## 2. Prerequisites and tools

### Prerequisites

You do not need to be an advanced mathematician or an experienced ML
engineer. You should be comfortable with:

-   Basic Python: variables, functions, loops, classes, lists,
    dictionaries, and modules.
-   Basic command-line usage and virtual environments.
-   Basic programming concepts such as functions, arrays, and debugging.

If your Python is rusty, revise it as needed while working through the
project. Avoid spending days studying unrelated Python libraries.

### Mathematics to learn as needed

Focus on the mathematics that directly supports neural networks:

-   Scalars, vectors, matrices, and tensors.
-   Matrix multiplication and transpose.
-   Dot products.
-   Probability and conditional probability.
-   Exponentials, logarithms, and softmax.
-   Derivatives, partial derivatives, and gradients.
-   The chain rule.

You do not need to complete a full linear algebra or calculus course
before starting. Learn the relevant ideas alongside the model components
that use them.

### Minimal technology stack

  -----------------------------------------------------------------------
  Tool                                Why it is used
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  PyTorch                             Tensors, automatic differentiation,
                                      neural-network layers, and training

  Matplotlib (optional)               Plotting training and validation
                                      loss

  A plain-text corpus                 Training data

  CPU or CUDA-capable GPU             Model training
  -----------------------------------------------------------------------

Install and learn only what the next stage requires. NumPy is useful,
but it is not a prerequisite for this project; PyTorch tensors are
sufficient for the implementation.

------------------------------------------------------------------------

## 3. Learning roadmap

Follow the stages in order. Each stage includes the concepts to
understand, a resource to use, and a checkpoint to complete before
moving on.

### Stage 1 --- Machine-learning foundations

**Learn**

-   What machine learning is and how it differs from rule-based
    programming.
-   Supervised, unsupervised, and self-supervised learning at a high
    level.
-   Training examples, inputs, targets, parameters, and hyperparameters.
-   Training, validation, and test data.
-   Model predictions and loss functions.
-   Generalization, overfitting, and underfitting.
-   The role of optimization.

**Resources**

-   [Neural Networks and Deep Learning --- Michael
    Nielsen](http://neuralnetworksanddeeplearning.com/), Chapters 1--2.
-   [Neural Networks: Zero to Hero --- Andrej
    Karpathy](https://karpathy.ai/zero-to-hero.html), introductory
    material.

**Checkpoint**

Explain how a model uses examples, a loss function, and parameter
updates to improve its predictions. Explain why training and validation
data serve different purposes.

### Stage 2 --- Neural networks and learning

**Learn**

-   Neurons, weights, and biases.
-   Linear transformations.
-   Activation functions: ReLU, sigmoid, and tanh.
-   Forward propagation.
-   Computational graphs.
-   Loss functions, especially cross-entropy.
-   Derivatives and the chain rule.
-   Backpropagation.
-   Gradient descent and learning rate.
-   Stochastic gradient descent and Adam.
-   The purpose of zeroing gradients during training.

**Resources**

-   [Zero to Hero --- micrograd / backpropagation
    material](https://karpathy.ai/zero-to-hero.html).
-   [PyTorch: Automatic Differentiation with
    `torch.autograd`](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
-   [Neural Networks and Deep
    Learning](http://neuralnetworksanddeeplearning.com/), selected
    sections on gradient descent and backpropagation.

**Checkpoint**

Trace a small network's forward pass, identify its loss, and explain how
backpropagation calculates gradients used by an optimizer to update
weights.

### Stage 3 --- Language modeling

**Learn**

-   What a language model estimates.
-   Tokens and sequences.
-   Probability of a sequence.
-   Conditional probability.
-   N-gram and bigram language models.
-   Next-token prediction.
-   Autoregressive modeling.
-   Cross-entropy loss for language modeling.
-   The difference between training and generation.
-   Perplexity as a language-model evaluation measure.

A central idea in autoregressive language modeling is:

\[ P(x_1, x_2, `\ldots`{=tex}, x_T) = `\prod`{=tex}*{t=1}\^{T} P(x_t
`\mid `{=tex}x_1, `\ldots`{=tex}, x*{t-1}) \]

The model estimates the next token from the tokens that came before it.

**Resources**

-   [Zero to Hero --- makemore
    material](https://karpathy.ai/zero-to-hero.html).
-   [GPT-2
    paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
    especially the language-modeling and training discussion.

**Checkpoint**

Explain how a model trained to predict one next token at a time can
generate a longer sequence by feeding each prediction back into the
model.

### Stage 4 --- Tokenization and embeddings

**Learn**

-   Why models operate on token IDs rather than raw text.
-   Character-level, word-level, and subword tokenization.
-   Vocabulary construction.
-   Encoding and decoding.
-   Byte Pair Encoding (BPE) at a conceptual level.
-   Embedding tables.
-   Token embeddings.
-   Positional information and positional embeddings.
-   Context length and sequence length.

For the first implementation, use a **character-level tokenizer**. It is
easy to inspect and avoids adding BPE implementation as another major
task. Learn BPE afterward as an extension.

**Resources**

-   [Let's Build the GPT Tokenizer --- Andrej
    Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE).
-   [Hugging Face Tokenizers
    documentation](https://huggingface.co/docs/tokenizers/index).
-   [The Illustrated Transformer --- Jay
    Alammar](https://jalammar.github.io/illustrated-transformer/), for
    positional information and the model's data flow.

**Checkpoint**

Take a short sentence and show how it becomes token IDs, how those IDs
index an embedding table, and how the model converts them into vectors.

### Stage 5 --- Attention

**Learn in this order**

1.  Why a sequence model needs a way to use information from earlier
    tokens.
2.  Self-attention.
3.  Queries, keys, and values.
4.  Query-key similarity scores.
5.  Scaling the scores.
6.  Softmax and attention weights.
7.  Weighted sums of values.
8.  Causal masking: preventing a position from seeing future tokens.
9.  Multi-head attention.
10. Concatenating heads and applying an output projection.

The scaled dot-product attention operation is:

\[ `\operatorname{Attention}`{=tex}(Q,K,V) =
`\operatorname{softmax}`{=tex}`\left`{=tex}(
`\frac{QK^\top}{\sqrt{d_k}}`{=tex} `\right`{=tex})V \]

Understand the role and shape of every term. In a decoder-only language
model, causal masking is essential: a token must not use information
from future target tokens during training.

**Resources**

-   [Let's Build GPT --- Andrej
    Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY), attention
    sections.
-   [The Illustrated
    Transformer](https://jalammar.github.io/illustrated-transformer/).
-   [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
    sections 3.2 and 3.2.2.

**Checkpoint**

Explain how queries, keys, and values produce attention-weighted
information. Explain why causal masking is required for autoregressive
generation.

### Stage 6 --- Transformer and GPT architecture

**Learn**

-   Encoder, decoder, and decoder-only architectures.
-   Why GPT uses a decoder-only Transformer.
-   Transformer blocks.
-   Causal multi-head self-attention.
-   Feed-forward networks / MLP blocks.
-   Residual connections.
-   Layer normalization.
-   Dropout.
-   Position information.
-   Vocabulary projection and logits.
-   Softmax probabilities.
-   Autoregressive sampling.

Understand the distinction:

  -----------------------------------------------------------------------
  Architecture            Typical example         Main purpose
  ----------------------- ----------------------- -----------------------
  Encoder-only            BERT                    Contextual
                                                  representations

  Decoder-only            GPT                     Autoregressive text
                                                  generation

  Encoder-decoder         Original Transformer,   Transform one sequence
                          T5                      into another
  -----------------------------------------------------------------------

A GPT-style block combines masked self-attention and a feed-forward
network, with residual connections and normalization. Exact ordering and
normalization variants differ between model implementations.

**Resources**

-   [Let's Build GPT --- Andrej
    Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY), complete
    lecture.
-   [The Annotated
    Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
-   [GPT-2
    paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

**Checkpoint**

Draw the data flow from token IDs to embeddings, through repeated
Transformer blocks, to vocabulary logits. Explain how the next-token
distribution is produced.

------------------------------------------------------------------------

## 4. Build the model

Once the concepts above are clear enough to explain, implement the
project in small, testable pieces. Do not begin by writing one large
model file.

### Project milestones

#### Milestone 1 --- Prepare the dataset

-   Obtain a plain-text corpus.
-   Read the text and inspect its contents.
-   Split the data into training and validation portions.
-   Build a vocabulary.
-   Implement text-to-token-ID encoding and token-ID-to-text decoding.
-   Check that encoding followed by decoding reproduces the original
    text.

#### Milestone 2 --- Create training sequences

-   Choose a context length (block size).
-   Create input sequences from the token IDs.
-   Create targets shifted one token to the right.
-   Build batches of input and target sequences.
-   Verify the shapes and alignment manually.

For example, if the input is:

`[a, b, c, d]`

the next-token targets are:

`[b, c, d, e]`

Each position is trained to predict the following token.

#### Milestone 3 --- Build a baseline language model

Before implementing the full Transformer, build a simple bigram model.

-   Map the current token to a distribution over the next token.
-   Calculate cross-entropy loss.
-   Train the model.
-   Generate a short sequence.

This gives you a baseline and makes the language-modeling objective
concrete.

#### Milestone 4 --- Implement embeddings and attention

-   Add token embeddings.
-   Add positional embeddings.
-   Implement scaled dot-product self-attention.
-   Add a causal mask.
-   Verify attention tensor shapes.
-   Extend the implementation to multiple heads.

Test each component with small inputs before combining them.

#### Milestone 5 --- Implement a Transformer block

Add:

-   Multi-head causal self-attention.
-   Residual connections.
-   Layer normalization.
-   Feed-forward network.
-   Dropout.

Verify that the block preserves the expected sequence and embedding
dimensions.

#### Milestone 6 --- Assemble the GPT-style model

-   Combine the embedding layers and Transformer blocks.
-   Project the final representations to vocabulary-sized logits.
-   Check output dimensions.
-   Calculate cross-entropy loss against the next-token targets.

#### Milestone 7 --- Train the model

A basic training iteration should follow this pattern:

1.  Get a batch of input and target sequences.
2.  Run the model to obtain logits.
3.  Calculate the loss.
4.  Clear old gradients.
5.  Backpropagate the loss.
6.  Update the parameters with an optimizer.
7.  Periodically evaluate on validation data.

Track both training and validation loss. A decreasing training loss
alone does not prove that the model generalizes.

#### Milestone 8 --- Generate text

-   Encode a prompt.
-   Pass the available context to the model.
-   Obtain logits for the next-token position.
-   Convert logits into probabilities.
-   Sample or select the next token.
-   Append the token to the context.
-   Repeat until the generation limit or stopping condition is reached.
-   Decode the generated token IDs into text.

Learn how temperature and top-k sampling affect generation after basic
greedy or multinomial sampling works.

------------------------------------------------------------------------

## 5. Recommended resources

The roadmap above assigns resources to the relevant stages. This section
collects the main references in one place.

### Video courses and lectures

  --------------------------------------------------------------------------------------------------------
  Resource                                                 Best for                Priority
  -------------------------------------------------------- ----------------------- -----------------------
  [Neural Networks: Zero to Hero --- Andrej                Backpropagation, neural Essential
  Karpathy](https://karpathy.ai/zero-to-hero.html)         networks, language      
                                                           modeling, and GPT       

  [Let's Build GPT --- Andrej                              Building a small        Essential
  Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)   GPT-style model step by 
                                                           step                    

  [Let's Build the GPT Tokenizer --- Andrej                Tokenization and BPE    Recommended
  Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)                           

  [Let's Reproduce GPT-2 (124M) --- Andrej                 A more advanced         Later
  Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)   training and            
                                                           reproduction workflow   
  --------------------------------------------------------------------------------------------------------

Do not try to complete every resource before coding. Use the material
that supports the stage you are currently studying.

### Documentation

  -----------------------------------------------------------------------------------------------------------------------
  Resource                                                                            Use it for
  ----------------------------------------------------------------------------------- -----------------------------------
  [PyTorch Tutorials](https://pytorch.org/tutorials/)                                 Tensors, datasets, models, and
                                                                                      training

  [PyTorch: Learn the                                                                 A guided introduction to PyTorch
  Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)                   

  [PyTorch                                                                            Automatic differentiation and
  Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)   gradients

  [PyTorch `nn` documentation](https://pytorch.org/docs/stable/nn.html)               Neural-network building blocks

  [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)             Tokenization concepts and tooling
  -----------------------------------------------------------------------------------------------------------------------

Use PyTorch's official documentation as the source of truth for API
behavior. The project does not require Hugging Face Transformers.

### Books and articles

  ---------------------------------------------------------------------------------------------------
  Resource                                                        Best sections
  --------------------------------------------------------------- -----------------------------------
  [Neural Networks and Deep Learning --- Michael                  Neural networks, gradient descent,
  Nielsen](http://neuralnetworksanddeeplearning.com/)             and backpropagation

  [Dive into Deep Learning](https://d2l.ai/)                      Linear neural networks, MLPs,
                                                                  attention, and Transformers

  [The Illustrated Transformer --- Jay                            Visual explanation of attention and
  Alammar](https://jalammar.github.io/illustrated-transformer/)   Transformer architecture

  [The Annotated Transformer --- Harvard                          A code-oriented Transformer
  NLP](https://nlp.seas.harvard.edu/annotated-transformer/)       walkthrough
  ---------------------------------------------------------------------------------------------------

These are reference materials, not books you must finish before
implementing the project.

### Research papers

Read the relevant sections rather than trying to digest every detail in
one sitting.

1.  **[Attention Is All You Need
    (2017)](https://arxiv.org/abs/1706.03762)**\
    The paper that introduced the Transformer architecture. Focus on
    attention, multi-head attention, positional encoding, and the
    feed-forward sublayers. Its original architecture is
    encoder-decoder, so do not confuse it with GPT's decoder-only
    design.

2.  **[Improving Language Understanding by Generative Pre-Training
    (GPT-1,
    2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**\
    Read for the idea of generative pretraining and how a Transformer
    language model can be adapted to downstream tasks.

3.  **[Language Models are Unsupervised Multitask Learners (GPT-2,
    2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**\
    Read for autoregressive language modeling, training, and the GPT-2
    approach.

### Code references

-   [nanoGPT --- Andrej Karpathy](https://github.com/karpathy/nanoGPT):
    a compact GPT training and implementation repository. Use it to
    inspect design choices and compare your implementation after
    attempting the components yourself.
-   [minGPT --- Andrej Karpathy](https://github.com/karpathy/minGPT): a
    small educational GPT implementation.

**Important:** Treat these repositories as references, not as a shortcut
to a finished project. First implement the components yourself; then
compare your design with a working implementation.

------------------------------------------------------------------------

## 6. Suggested first experiment

Keep the first model deliberately small. The purpose is to understand
the complete training and generation pipeline, not to maximize output
quality.

### Dataset

Start with the [Tiny Shakespeare text
dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

It is a small character-level text corpus that works well for an
educational language-model experiment.

### Initial model configuration

These are illustrative starting values, not universal recommendations.

  Setting                             Starting value
  ----------------------- --------------------------
  Tokenization                       Character-level
  Context length                                 128
  Embedding dimension                            128
  Attention heads                                  4
  Transformer blocks                               4
  Dropout                                        0.1
  Weight initialization                       Random
  Architecture              Decoder-only Transformer

Adjust the model size and batch size to your available hardware. A CPU
can run a small experiment, but training may be slow. A CUDA-capable GPU
can make iteration faster.

### What success looks like

A successful first experiment means:

-   The dataset loads and tokenizes correctly.
-   Input and target sequences are aligned.
-   The model produces logits with the expected shape.
-   The loss is finite and can be backpropagated.
-   Parameters update during training.
-   Training and validation loss can be measured.
-   The model generates text from a prompt using its own learned
    weights.
-   You can explain the role of each major component.

The generated text may be repetitive, grammatically incorrect, or
nonsensical. That is expected for a small model trained on a limited
dataset. It does not invalidate the implementation.

------------------------------------------------------------------------

## 7. What to defer

These subjects are useful in broader ML and AI engineering, but they are
not prerequisites for this first GPT-style model.

-   NumPy in depth.
-   Pandas and general data-analysis workflows.
-   Classical ML algorithms such as SVMs, random forests, and gradient
    boosting.
-   CNNs and image-processing architectures.
-   Full implementations of RNNs and LSTMs.
-   Reinforcement learning.
-   GANs and diffusion models.
-   Distributed training and multi-GPU optimization.
-   CUDA kernel development.
-   Large-scale data pipelines.
-   RAG frameworks, vector databases, and agent frameworks.
-   Fine-tuning and deployment infrastructure.

Learn these later when you have a reason to use them. Do not mistake a
broad AI syllabus for a prerequisite checklist.

------------------------------------------------------------------------

## 8. Completion checklist

Use this checklist to judge your understanding and project progress.

### Foundations

-   [ ] I can explain parameters, loss, training, and validation.
-   [ ] I understand vectors, matrices, dot products, and tensor shapes.
-   [ ] I can explain forward propagation and backpropagation.
-   [ ] I understand gradient descent and the role of an optimizer.

### Language modeling

-   [ ] I can explain conditional probability and next-token prediction.
-   [ ] I understand autoregressive generation.
-   [ ] I can explain cross-entropy loss for next-token prediction.
-   [ ] I understand the purpose of tokenization and embeddings.

### Transformer

-   [ ] I can explain queries, keys, and values.
-   [ ] I understand scaled dot-product attention.
-   [ ] I can explain causal masking.
-   [ ] I understand multi-head attention.
-   [ ] I can explain residual connections, normalization, and
    feed-forward layers.
-   [ ] I can draw the flow of a decoder-only Transformer.

### Implementation

-   [ ] I implemented a tokenizer and verified encoding/decoding.
-   [ ] I created correctly shifted input and target sequences.
-   [ ] I trained a simple baseline language model.
-   [ ] I implemented causal self-attention.
-   [ ] I assembled a decoder-only Transformer.
-   [ ] I trained it from randomly initialized weights.
-   [ ] I tracked training and validation loss.
-   [ ] I generated text from a prompt.
-   [ ] I can explain the code without relying on a tutorial.

------------------------------------------------------------------------

## 9. How to use AI as a tutor without outsourcing the work

If you use an AI assistant while following this guide, keep it in the
role of a tutor rather than a code generator.

Good uses:

-   Ask for an explanation of a concept in plain language.
-   Ask for a mathematical derivation, one step at a time.
-   Ask what the dimensions of a tensor should be at a particular point.
-   Ask for a small conceptual example.
-   Ask questions that test your understanding.
-   Share your own code and ask for help understanding an error.
-   Ask for hints before asking for a complete explanation.

Avoid asking it to:

-   Generate the entire model.
-   Write each implementation stage for you.
-   Produce a finished training loop before you understand it.
-   Hide the mathematics behind a library call.

A useful prompt:

> Act as an ML instructor. Teach me this concept from first principles.
> Explain the problem it solves, build the intuition, introduce the
> mathematics gradually, and walk through a small example. Explain the
> relevant tensor shapes. Do not generate implementation code unless I
> explicitly ask for a tiny illustrative example. End by asking me
> questions that test whether I actually understand the concept.

The most important part of this project is not getting a model to
produce text. It is understanding why the model produces that text and
being able to build the system again without blindly following someone
else's implementation.

------------------------------------------------------------------------

## Final objective

Build a small language model whose architecture, training process, and
generation loop you understand.

Start with a simple language model, progress to attention, implement the
Transformer, and train it on text. Keep the first version small and
understandable. Improve it only after the complete pipeline works.

**Understand the components. Implement them yourself. Test each stage.
Then put them together.**
