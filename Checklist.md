# Checklist

## Data

* [X]  Scrape raw text
* [X]  Auotomatic data annotation @Xiaotong
* [X]  Determine data size @Lexa

### Data annotation quality check @Lexa @Xiaotong @Sean
* [ ]  Manually check the quality of generated data annotation
* [ ]  *Split trian/test data (TBD)*
* [ ]  Randomly extract 10 QA from **test set** (6*10), compute IAA and report findings


## Pipeline

All basic pipeline is done. The following items are advanced options to be discussed.

### Data Store

* [ ]  Semantic chunker length issue: 33/8156 chunks exceed the limit of 40960 bytes per vector. Use `RecursiveCharacterTextSplitter` to sub-split
1. Only sub-split the chunks exceeding limits (still have long text)
2. Sub-split length outliers
* [X]  Whether to chunk raw_data and preformatted_data separately?
* [ ]  Sentence-generation model options for embedding

### Retriever

* [ ]  Reranking
* [ ]  Hyperparameters for `PineconeVectorStore.as_retriever()`

### QA Generator

* [ ]  Prompt template
* [ ]  Hyperparameters for `pipeline()`
* [ ]  Text-generation model options

## Experiments

* [ ]  Define evaluation function
* [ ]  Write evaluation pipeline

## Submission

* [ ]  TBD
