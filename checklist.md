# Checklist

## Data

* [X]  Scrape raw text
* [X]  Automatic data annotation @Xiaotong
* [X]  Determine data size @Lexa

### Data Annotation Quality Check

Manually check the quality of generated data annotation.

* [X]  (1) `Annotated_dataeventscmu.txt` @Lexa
* [X]  (1) `Annotated_dataeventspittsburgh.txt` @Sean
* [ ]  (1) `Annotated_datageneralinfo_cmu.txt` @Xiaotong
* [ ]  (1) `Annotated_datageneralinfo_pittsburgh.txt`@Xiaotong
* [X]  (1) `Annotated_datamusicculture.txt` @Lexa
* [X]  (1) `Annotated_datasports.txt` @Lexa

### Dataset
* [ ]  (2) Split train/test data: 200-300 test samples  @Xiaotong
* [ ]  (3) Randomly extract 50 samples from the test set  @Xiaotong
* [ ]  (4) Compute IAA  @Sean @Lexa

## Pipeline

All basic pipeline is done. The following items are advanced options to be discussed.

### Data Store

* [ ]  (2) Don't chunk raw_data and preformatted_data separately?  @Sean
* [ ]  (2) Semantic chunker length issue: 33/8156 chunks exceed the limit of 40960 bytes per vector. Use `RecursiveCharacterTextSplitter` to sub-split using length outliers.  @Sean
* [ ]  (2) `sentence-generation` model options for embedding. See the [leaderboard](https://sbert.net/docs/sentence_transformer/pretrained_models.html) for reference. We currently use `all-mpnet-base-v2`.  @Lexa

### Retriever

* [ ]  (2) Hyperparameters for `PineconeVectorStore.as_retriever()`  @Sean
* [ ]  Reranking (TBD)

### QA Generator

* [ ]  (2) Prompt template  @Xiaotong
* [ ]  (2) Hyperparameters for `pipeline()`  @Sean
* [ ]  (2) `transformer` model options for rag generator. See [Hugging Face model page](https://huggingface.co/models) for reference. Models should support `"text2text-generation"` tasks. We currently use `google/flan-t5-large`  @Lexa

## Experiments

* [ ]  (3) Model variant experiment pipeline  @Xiaotong
* [ ]  (3) Performance evaluation pipeline  @Lexa

## Submission

* [ ]  TBD
