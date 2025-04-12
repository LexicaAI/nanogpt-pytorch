# NanoGPT

A lightweight implementation of the GPT (Generative Pre-trained Transformer) model with a clean, modular architecture.

## Features

- 🧠 Clean implementation of GPT-2 architecture
- 🚀 Training with FineWeb-Edu dataset
- 📊 Evaluation on HellaSwag benchmark
- 🔄 Support for distributed training
- 💾 Checkpointing and model loading
- 🛠️ Flexible configuration options

## Installation

Clone this repository:

```bash
git clone https://github.com/LexicaAI/nanogpt-pytorch.git
cd nanogpt-pytorch
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a GPT model from scratch:

```bash
python scripts/train.py --batch_size 32 --steps 10000
```

Fine-tune a pre-trained GPT-2 model:

```bash
python scripts/train.py --model_type gpt2 --batch_size 32 --steps 5000
```

### Evaluation

Evaluate a trained model on the HellaSwag benchmark:

```bash
python scripts/evaluate.py --task hellaswag
```

Measure perplexity on the validation set:

```bash
python scripts/evaluate.py --task perplexity --split val
```

### Text Generation

Generate text with a trained model:

```bash
python scripts/generate.py --prompt "Once upon a time" --max_tokens 100 --temperature 0.8
```

## Project Structure

The repository is organized as follows:

```
nanogpt/
├── nanogpt/                      # Main package
│   ├── model/                    # Model architectures
│   │   ├── attention.py          # Attention mechanisms
│   │   ├── mlp.py                # MLP module
│   │   ├── block.py              # Transformer block
│   │   └── gpt.py                # GPT model
│   ├── tokenizer/                # Tokenizer architectures
│   │   ├── tokenizer.py          # Tokenization utilities
│   ├── data/                     # Data processing modules
│   │   ├── dataloader.py         # DataLoader implementation
│   │   ├── datasets/             # Dataset-specific processors
│   │   │   ├── hellaswag.py      # HellaSwag dataset
│   │   │   └── fineweb.py        # FineWeb dataset
│   ├── training/                 # Training utilities
│   │   ├── config.py             # GPT-2 specific configuration
│   │   ├── trainer.py            # Training loop implementation
│   │   └── optimizer.py          # Optimization utilities
│   ├── evaluation/               # Evaluation utilities
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── evaluator.py          # Evaluation pipeline
│   └── utils/                    # Utility functions
│       ├── distributed.py        # Distributed training utilities
│       ├── checkpoint.py         # Model checkpointing
│       └── logging.py            # Logging utilities
└── scripts/                      # Executable scripts
    ├── train.py                  # Training script
    ├── evaluate.py               # Evaluation script
    └── generate.py               # Text generation script
```

## Customization

The implementation is designed to be modular and easily customizable. You can modify any component to adapt it to your specific needs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is inspired by:
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [OpenAI GPT-2](https://github.com/openai/gpt-2)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) 