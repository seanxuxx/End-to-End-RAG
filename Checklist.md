# Checklist

## Data

* [X]  Scrape raw text
* [X]  Automatic data annotation @Xiaotong
* [X]  Determine data size @Lexa

### Data annotation quality check 
* [ ]  (1) **Manually check the quality of generated data annotation**  @Lexa @Xiaotong @Sean
* [ ]  (2) Split train/test data: 200-300 test samples  @Xiaotong
* [ ]  (3) Randomly extract 50 samples from test set  @Xiaotong
* [ ]  (4) Compute IAA @Sean @Lexa

## Pipeline

All basic pipeline is done. The following items are advanced options to be discussed.

### Data Store

* [X]  Whether to chunk raw_data and preformatted_data separately?  @Sean
* [ ]  (1) **Semantic chunker length issue: 33/8156 chunks exceed the limit of 40960 bytes per vector. Use `RecursiveCharacterTextSplitter` to sub-split using length outliers**  @Sean
* [ ]  (2) Sentence-generation model options for embedding  @Lexa

### Retriever

* [ ] (1) **Hyperparameters for `PineconeVectorStore.as_retriever()`**  @Sean
* [ ]  Reranking (TBD)

### QA Generator

* [ ]  (2) Prompt template  @Xiaotong
* [ ]  (1) **Hyperparameters for `pipeline()`**  @Sean
* [ ]  (2) Text-generation model options  @Lexa

## Experiments

* [ ]  (3) Config experiment pipeline  @Xiaotong
* [ ]  (3) Performance evaluation pipeline  @Lexa

## Submission

* [ ]  TBD
