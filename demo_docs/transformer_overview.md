# Transformer Architecture — Technical Overview

## Background

The Transformer is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google Brain. It replaced recurrent neural networks (RNNs and LSTMs) as the dominant architecture for sequence modelling tasks and became the foundation for virtually every large language model developed since 2018.

The core insight of the Transformer is that attention mechanisms alone, without any recurrence or convolution, are sufficient to capture dependencies across arbitrarily long sequences — and can do so in parallel, making training dramatically faster.

## Self-Attention

The central operation in the Transformer is scaled dot-product attention. Given an input sequence of token embeddings, three linear projections produce queries (Q), keys (K), and values (V). Attention scores are computed as:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The scaling factor sqrt(d_k) prevents the dot products from growing too large in magnitude, which would push the softmax into saturation regions with near-zero gradients.

Each token attends to all other tokens simultaneously (hence "self-attention"), producing a weighted sum of values where the weights reflect relevance. This replaces the sequential hidden state passing of RNNs and enables parallelisation across the sequence length dimension.

## Multi-Head Attention

Rather than computing a single attention function, the Transformer uses multi-head attention: h separate attention heads, each with its own Q, K, V projection matrices. The outputs are concatenated and projected:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

Multiple heads allow the model to attend to different positions for different reasons simultaneously — one head might capture syntactic dependencies, another coreference, another positional proximity.

## Architecture

The Transformer follows an encoder-decoder structure for sequence-to-sequence tasks (e.g. translation). Each encoder layer contains:
1. Multi-head self-attention
2. Position-wise feed-forward network (two linear layers with ReLU)
3. Residual connections and layer normalisation around each sub-layer

The decoder adds a third sub-layer: cross-attention over the encoder output, allowing the decoder to attend to all encoder positions when generating each output token.

For language modelling and retrieval tasks, only the encoder stack is used (BERT-style) or only the decoder (GPT-style).

## Positional Encoding

Transformers have no inherent notion of order — attention is permutation-equivariant. Positional encodings inject position information by adding fixed or learned vectors to the input embeddings. The original paper used sinusoidal encodings of different frequencies, enabling the model to generalise to sequence lengths not seen during training.

Modern models use Rotary Position Embedding (RoPE) or ALiBi, which inject relative position information directly into the attention score computation.

## Pre-training and Fine-tuning

The Transformer's impact was amplified by the pre-training paradigm. Models pre-trained on large corpora learn general language representations that transfer to downstream tasks through fine-tuning on task-specific data.

**BERT** (Bidirectional Encoder Representations from Transformers, 2018): encoder-only, pre-trained with masked language modelling (MLM) and next sentence prediction (NSP). Excellent for classification, NER, retrieval.

**GPT** (Generative Pre-trained Transformer, 2018-): decoder-only, pre-trained with causal language modelling (predict next token). Excellent for generation.

**T5** (Text-to-Text Transfer Transformer, 2019): encoder-decoder, frames all tasks as text-to-text.

## Scaling Laws

Kaplan et al. (OpenAI, 2020) established empirical scaling laws showing that model performance improves predictably as a power law with model size, dataset size, and compute budget. This gave the field a principled basis for investing in larger models and drove the development of GPT-3, PaLM, Llama, and subsequent frontier models.

The key finding: given a fixed compute budget, it is better to train a larger model on fewer tokens than a smaller model to convergence. Hoffmann et al. (DeepMind, 2022) later refined this with the Chinchilla scaling laws, suggesting models should be trained on roughly 20 tokens per parameter.

## Relation to Retrieval Models

Bi-encoder retrieval models (used in RAG systems) are typically BERT-style Transformer encoders fine-tuned with contrastive objectives (e.g. MultipleNegativesRankingLoss) to map semantically similar texts close together in embedding space.

Cross-encoder rerankers are also BERT-style encoders but take concatenated (query, document) pairs as input and output a relevance score. Because both inputs are processed jointly, cross-encoders can model fine-grained query-document interactions that bi-encoders miss — at the cost of quadratic inference complexity relative to the number of candidates.