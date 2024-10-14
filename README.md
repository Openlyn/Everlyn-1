Visit our website: [Everlyn.ai](https://www.everlyn.ai)

## 🔆 Research Overview

Our latest open-source research is centered around three key components. First, we introduce a new approach to video compression and tokenization, designed to improve both quality and performance. Next, we present our framework for efficient autoregressive models. Finally, we share our advancements in multimodal understanding, with a focus on reducing hallucinations in large language models.

### 1. [Distribution Matching for Vector Quantization](https://github.com/Openlyn/Wasserstein-VQ)

We tackle the challenges of instability and inefficiency in vector quantization for autoregressive video models. By employing a novel distribution matching approach based on the **Wasserstein distance**, we significantly enhance codebook utilization and reduce quantization errors. This method results in more stable training and improved performance in generative video tasks.

### 2. [EfficientARV: Efficient Autoregressive Models for Image and Video Generation](https://github.com/Openlyn/EfficientARV)

EfficientARV is designed to create an efficient autoregressive model for jointly generating images and videos. The project explores multiple conditional generation tasks, such as image animation, inpainting, outpainting, video prediction, and video interpolation. Additionally, it aims to integrate these generation capabilities into Multimodal Large Language Models (MLLMs) for more interactive and robust AI systems.

### 3. [ANTRP: Intervening Anchor Token - Decoding Strategy for MLLMs](https://github.com/Openlyn/ANTRP)

Lastly, we focus on improving Multimodal Large Language Models (MLLMs) by addressing the hallucination problem. Instead of penalizing summary tokens, ANTRP intervenes in the query-key parameters variance, reducing hallucinations without additional inference time. We propose the **Dynamic Token Propagation Mechanism (TAME)**, which dynamically adjusts the eigenspectrum variance of the attention weights to alleviate over-propagation of "anchor" tokens. Extensive experiments show a strong correlation between the eigenspectrum and hallucinations, with TAME significantly reducing hallucinated objects across various MLLMs.

---

At Everlyn, we continue to develop AI that pushes the boundaries of what's possible in video generation, turning the dream of limitless video AI into a reality.
