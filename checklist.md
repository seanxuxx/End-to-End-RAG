# Checklist

## Data

* [X]  Scrape raw text
* [X]  Automatic data annotation @Xiaotong
* [X]  Determine data size @Lexa

### Data Annotation Quality Check

Manually check the quality of generated data annotation.

* [X]  (1) `Annotated_dataeventscmu.txt` @Lexa
* [X]  (1) `Annotated_dataeventspittsburgh.txt` @Sean
* [X]  (1) `Annotated_datageneralinfo_cmu.txt` @Xiaotong
* [X]  (1) `Annotated_datageneralinfo_pittsburgh.txt`@Xiaotong
* [X]  (1) `Annotated_datamusicculture.txt` @Lexa
* [X]  (1) `Annotated_datasports.txt` @Lexa

### Dataset
* [X]  (2) Split train/test data: 200-300 test samples  @Xiaotong
* [X]  (3) Randomly extract 50 samples from the test set  @Xiaotong
* [ ]  (4) Compute IAA  @Sean @Lexa

## Pipeline

All basic pipeline is done. The following items are advanced options to be discussed.

### Data Store

* [X]  (2) Don't chunk raw_data and preformatted_data separately?  @Sean
* [X]  (2) Semantic chunker length issue: 33/8156 chunks exceed the limit of 40960 bytes per vector. Use `RecursiveCharacterTextSplitter` to sub-split using length outliers.  @Sean
* [ ]  (2) `sentence-generation` model options for embedding. See the [leaderboard](https://sbert.net/docs/sentence_transformer/pretrained_models.html) for reference. We currently use `all-mpnet-base-v2`.  @Lexa

### Retriever

* [X]  (2) Add hyperparameter args for `PineconeVectorStore.as_retriever()`  @Sean
* [ ]  Reranking (TBD)

### QA Generator

* [ ]  (2) Prompt template  @Xiaotong
* [X]  (2) Add hyperparameter args for `pipeline()`  @Sean
* [ ]  (2) `transformer` model options for rag generator. See [Hugging Face model page](https://huggingface.co/models) for reference. Models should support either `"text-generation"` or `"text2text-generation"` tasks. We currently use `"mistralai/Mistral-7B-Instruct-v0.2"` for `"text-generation"` and `google/flan-t5-large` for `"text2text-generation"`.   @Lexa

## Experiments

* [ ]  (3) Model variant experiment pipeline  @Xiaotong
* [ ]  (3) Performance evaluation pipeline  @Lexa

## Submission

* [ ]  TBD
