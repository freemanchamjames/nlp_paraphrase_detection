# nlp_paraphrase_detection

DASC 5931 Natural Language Processing project using the Microsoft Research Paraphrase Corpus (MRPC).

## Research Question

How many labeled sentence pairs does a fine-tuned transformer need before it passes a prompted LLM at judging whether two sentences mean the same thing? And do both systems fail on the same pairs: high word overlap but different meaning, or low overlap but the same meaning?

## Dataset

Microsoft Research Paraphrase Corpus (MRPC)

- Train: 3,668 pairs
- Validation: 408 pairs
- Test: 1,725 pairs

## Project Structure

- `notebooks/` - exploration and experiments
- `src/` - reusable Python code
- `data/` - local data files
- `results/` - figures, tables, and metrics
- `reports/` - project reports
- `models/` - saved model files
