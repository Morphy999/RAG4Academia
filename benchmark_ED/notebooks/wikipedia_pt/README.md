---
dataset_info:
  features:
  - name: id
    dtype: large_string
  - name: url
    dtype: large_string
  - name: title
    dtype: large_string
  - name: chunk
    dtype: large_string
  - name: chunk_number
    dtype: int64
  - name: embeddings
    large_list: float64
  splits:
  - name: train
    num_bytes: 7120517350
    num_examples: 2052058
  download_size: 5436210633
  dataset_size: 7120517350
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: eupl-1.1
language:
- pt
---
# Dataset Card for Dataset Name

<!-- Provide a quick summary of the dataset. -->

A [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) using only the portuguese subset. An embeddings column is added to enable vector search. 

The dataset has been chunked using [chonkie](https://github.com/chonkie-ai/chonkie) and [sentence transformers](https://www.sbert.net/) (model: [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1))

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->



- **Curated by:** [marquesafonso](https://huggingface.co/marquesafonso)
- **Language(s) (NLP):** Portuguese
- **License:** eupl-1.1

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)

## Uses

Vector search over wikipedia in portuguese.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

[More Information Needed]

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

[More Information Needed]

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

Enabling vector search over the wikipedia corpus, namely its portuguese subset.

### Source Data

Wikipedia data as per the following [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia).

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

[More Information Needed]

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

[More Information Needed]

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

[More Information Needed]

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

[More Information Needed]

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

[More Information Needed]

## Dataset Card Contact

[More Information Needed]