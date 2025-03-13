# Checklist

## Data

* [X]  Scrape raw text
* [X]  **Auotomatic data annotation** @Xiaotong
* [ ]  Determine data size @Lexa
* [ ]  Mannual data annotation @Lexa
* [ ]  *Split trian/test data (TBD)*
* [ ]  *Measure IAA*

## Pipeline

All basic pipeline is done. The following items are advanced options to be discussed.

### Data Store

* [ ]  Which chunker options to include? Hyperparameters for chunking
* [ ]  Whether to chunk raw_data and preformatted_data separatedly?
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
