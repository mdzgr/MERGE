# MERGE: Minimal Expression-Replacement GEneralization Test for Natural Language Inference

Repository for the paper **MERGE: Minimal Expression-Replacement GEneralization Test for Natural Language Inference**, accepted at **BlackboxNLP 2026**.

MERGE is a methodology for generating variants of NLI problems by replacing shared words of the premise and the hypothesis. MERGE evaluates models on these variants, with consistency tests showing requiring evaluated models to get more than ~50% of variants correct results in decreased performance.

## Repository contents

- **demo_notebook.ipynb:** Choose a premise–hypothesis pair from SNLI or MNLI, 2 masked language models, a part of speech to replace, and the notebook generates a number of variants for you.
- - **`setup.demo.py`:** Utils required for the notebook.
- **`demo.py`:** Functions used for the experiemnts from identifying shared words, obtaining replacement suggestions from language models, filtering suggestions, to generating variants.
- **`utils.py`:** Functions for data processing and evaluation.
- **variants(`.zip`):** Archive of the generated variants used in the paper.


## Run the demo in Google Colab

1. To run the demo, acess [demo notebook](https://colab.research.google.com/drive/1fBJV2a6zJPUd7QU1QE04LF0mB697ogtk?usp=sharing).
2. Follow the dataset download instructions in the notebook.

To generate variants for the demo notebook you will need to download the required source datasets and place them in an `nlp_data` folder in your Google Drive.

