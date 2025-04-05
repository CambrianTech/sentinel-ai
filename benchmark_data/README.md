# Benchmark Datasets for Sentinel-AI

This directory contains datasets used for benchmarking model performance, particularly for pruning experiments. The benchmark system supports both synthetic data and real data from various sources.

## Available Datasets

### Project Gutenberg Books

Real literary text data from classic novels, including:
- Pride and Prejudice by Jane Austen
- The Adventures of Sherlock Holmes by Sir Arthur Conan Doyle
- The Count of Monte Cristo by Alexandre Dumas

These datasets provide high-quality, diverse natural language for realistic benchmarking scenarios.

### Processed Datasets

The benchmark system creates processed datasets from the raw sources, with:
- Cleaned text (removed headers, footers, etc.)
- Segmented into appropriate lengths for model training
- Separate train/validation splits
- Deduplicated content

These processed datasets are automatically saved and can be reused in future benchmark runs.

### Synthetic Datasets

When real data is not available, the system falls back to high-quality synthetic data:
- AI, ML, and NLP-focused Wikipedia-style articles
- Literary excerpts from famous authors
- Scientific articles on physics, mathematics, and computer science

## Using the Datasets

To use these datasets in benchmarking, run:

```bash
python scripts/benchmark_with_metrics.py \
  --model_name distilgpt2 \
  --output_dir ./benchmark_results \
  --eval_dataset "gutenberg" \
  --use_real_data \
  --eval_samples 500
```

## Dataset Options

| Dataset Key | Description |
|-------------|-------------|
| `gutenberg`, `books`, `classics` | All Project Gutenberg books |
| `pride`, `austen` | Pride and Prejudice only |
| `sherlock`, `holmes` | Sherlock Holmes only |
| `monte`, `cristo`, `dumas` | Count of Monte Cristo only |
| `processed`, `gutenberg_processed` | Pre-processed datasets from previous runs |

## Adding New Datasets

To add new books:
1. Download text files from [Project Gutenberg](https://www.gutenberg.org/)
2. Place them in the `benchmark_data/gutenberg/` directory
3. They will be automatically detected and used

For other dataset types:
1. Add the dataset file or directory to this folder
2. Update the data paths in `scripts/benchmark_with_metrics.py`