# AI-Powered Headline Clickbait Detection and Scoring

This project compares two models for classifying English headlines as **dataset-labeled clickbait** (`1`) or **non-clickbait** (`0`): a TF–IDF + Logistic Regression baseline and a fine-tuned DistilBERT model. It includes data preparation, controlled DistilBERT experiments, final test-set evaluation, and a simple interactive notebook demo.

> The clickbait label describes a headline style according to the source datasets. It does not mean the headline is false or that the article contains misinformation. The displayed clickbait probability is model confidence, not a verified severity or misleadingness score.

## Project structure

```text
csc173-midterm/
├── datasets/
│   ├── raw/                 # Original Kaggle CSVs
│   └── processed/           # Cleaned data, shared splits, conflict report
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_distilbert_finetuning.ipynb
│   ├── 04_test_evaluation.ipynb
│   └── 05_demo.ipynb
├── results/
│   ├── preliminary/         # Earlier validation figures
│   └── final_test/          # Final test metrics and predictions
└── README.md
```

The combined cleaned dataset contains **50,261 headlines**. The shared stratified split is **40,208 training**, **5,027 validation**, and **5,026 test** examples (80/10/10). Dataset 2's `other` label is excluded; `news` maps to `0` and `clickbait` maps to `1`. Mojibake and whitespace issues, missing values, duplicates, and label conflicts are checked during preparation.

## How to run

Google Colab is recommended for DistilBERT training when a GPU is available. The notebooks can be run independently in Colab; the saved CSVs, metrics, and model files pass results between them.

1. Run `01_data_prep.ipynb` only if you need to regenerate the cleaned data and fixed splits. Existing processed CSVs can be reused.
2. Run `02_tfidf_baseline.ipynb` to train the TF–IDF + Logistic Regression baseline and evaluate it on validation. Save the fitted pipeline (`tfidf_lr.joblib`) for the demo.
3. Run `03_distilbert_finetuning.ipynb` to train five DistilBERT configurations on the same training split and compare them on validation. The selected model is the run with the highest clickbait F1 (V3 in the recorded tuning comparison). Set the selected model folder consistently in later notebooks.
4. Run `04_test_evaluation.ipynb` once model selection is complete. It loads the chosen DistilBERT model from Google Drive and prompts you to upload the TF–IDF `.joblib` file if needed, then evaluates both models on the same test split.
5. Run `05_demo.ipynb` for an interactive prediction demo with a text box, model selector, and Predict button.

Keep the same split files for every run. Use training data to fit models, validation data to compare configurations, and test data for the final comparison. The test set was accessed earlier during project development, so interpret the final test metrics with that caveat. Model weights are not committed to this repository; save them in Google Drive for reuse. The notebooks expect the shared CSVs in the configured data directory; check their path settings if your local folder layout differs.

## DistilBERT tuning results (validation)

Five configurations were compared on the same validation set. Clickbait is the positive class. V3 had the highest clickbait F1, but the scores were very close; the tuning did not produce a large improvement over V1.

| Version | Epochs | Learning rate | Accuracy | Clickbait precision | Clickbait recall | Clickbait F1 | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | 2 | 1e-5 | 92.08% | 91.20% | 88.33% | **89.74%** | 398 |
| V1 | 2 | 2e-5 | 92.04% | 90.81% | 88.69% | 89.73% | 400 |
| V4 | 2 | 3e-5 | 91.94% | 90.15% | 89.19% | 89.67% | 405 |
| V5 | 3 | 1e-5 | 91.98% | 91.00% | 88.28% | 89.62% | 403 |
| V2 | 3 | 2e-5 | 91.47% | 89.30% | 88.89% | 89.09% | 429 |

## Final test-set results

The selected DistilBERT model (`v3`) and the unchanged TF–IDF baseline were evaluated on the same **5,026-headline** test split. Clickbait precision, recall, and F1 refer to class `1`.

| Model | Accuracy | Clickbait precision | Clickbait recall | Clickbait F1 | TN | FP | FN | TP | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF–IDF + Logistic Regression | 88.28% | 88.51% | 80.56% | 84.35% | 2,850 | 206 | 383 | 1,587 | 589 |
| DistilBERT (`v3`) | **90.69%** | **90.25%** | **85.48%** | **87.80%** | **2,874** | **182** | **286** | **1,684** | **468** |

DistilBERT had higher scores on all four reported metrics in this test run, with 121 fewer total errors. These results compare the models on this dataset and split; they do not establish that contextual understanding alone caused the difference.

### Per-class test report

| Model / class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| TF–IDF / non-clickbait | 88.15% | 93.26% | 90.63% | 3,056 |
| TF–IDF / clickbait | 88.51% | 80.56% | 84.35% | 1,970 |
| DistilBERT / non-clickbait | 90.95% | 94.04% | 92.47% | 3,056 |
| DistilBERT / clickbait | 90.25% | 85.48% | 87.80% | 1,970 |

### Selected DistilBERT test examples

These examples show the dataset label (`actual`) and the selected DistilBERT model's prediction. The labels are dataset annotations, not an independent assessment of truthfulness.

| Outcome | Headline | Actual | Predicted | Clickbait confidence |
|---|---|---|---|---:|
| True clickbait (true positive) | 7 Easy Dinners To Make This Week | Clickbait | Clickbait | 99.51% |
| True non-clickbait (true negative) | The Rush to Stake a Domain Claim | Non-clickbait | Non-clickbait | 1.03% |
| False negative (missed clickbait) | BBC Presenter Given Warning after Saying Trans Rights | Clickbait | Non-clickbait | 34.47% |
| False positive (flagged as clickbait) | The Surprising Musical Preferences of an Amazo... | Non-clickbait | Clickbait | 58.14% |

## Technologies used

- **Python** for data processing, training, and evaluation
- **Pandas / NumPy** for dataset handling
- **Scikit-learn** for stratified splits, TF–IDF, Logistic Regression, and metrics
- **Hugging Face Transformers and Datasets** for DistilBERT tokenization and fine-tuning
- **PyTorch** for deep-learning execution
- **Matplotlib and Seaborn** for metric and confusion-matrix visualizations
- **Joblib** for saving/loading the TF–IDF pipeline
- **Google Colab and Google Drive** for GPU training and model storage
- **Kaggle** as the dataset source

## Limitations

- The models reproduce the datasets' clickbait labeling conventions; they do not determine whether a headline or article is true, false, or misleading.
- The models analyze headline text only and do not read the linked article or verify claims.
- The datasets are primarily English, so use is recommended for English headlines; performance on other languages is unknown.
- The confidence percentage is a model probability estimate and is not a calibrated severity score.
- Results depend on these datasets and the fixed split; they may differ on headlines from other publishers or time periods.

## References

- Alarfaj, F. K., Muqadas, A., Khan, H. U., & Naz, A. (2025). Clickbait detection in news headlines using Roberta-large language model and deep embeddings. *Scientific Reports, 16*(1). [https://doi.org/10.1038/s41598-025-30229-5](https://doi.org/10.1038/s41598-025-30229-5)
- Dewangga, R. T., & Prasetiyo, B. (2025). Comparative analysis of BERT model and DistilBERT model for enhanced clickbait headline structure detection in Indonesian online news. *JUITA: Jurnal Informatika*, 235–244. [https://doi.org/10.30595/juita.v13i3.26479](https://doi.org/10.30595/juita.v13i3.26479)
- Mohamed, M., Youssef, Y., & Nassar, S. (2026). From keywords to context: Comparing TF–IDF and DistilBERT for clickbait classification. *Damanhour Journal of Intelligent Systems and Informatics, 4*(1), 0–0. [https://doi.org/10.21608/djis.2026.508137.1027](https://doi.org/10.21608/djis.2026.508137.1027)

## Project members

- Bryan Miguel Agan
- Rogelio Angelo Bollozos
- Marcus Miguel Cuaton
