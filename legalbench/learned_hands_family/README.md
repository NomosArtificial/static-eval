# learned_hands_family

**Contributor**: Legal Design Lab

**Source**: [Learned Hands](https://spot.suffolklitlab.org/data/#learnedhands)

**License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Task summary**: Classify if a user post implicates legal isssues related to family.

**Size (samples)**: 7567

## Task Description

This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.

## Task Construction

This task was constructed from the [LearnedHands](https://suffolklitlab.org/) dataset. Please see their website for more information on annotation. Our task consists of a binarized version of the original dataset, with "negatives" randomly sampled from posts with other topics. This dataset is class balanced.

## Column names

- `text`: user post
- `answer`: class label (Yes/No)