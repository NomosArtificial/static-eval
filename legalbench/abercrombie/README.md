# abercrombie

**Contributor**: Neel Guha
 
**Source**: Neel Guha
 
**License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
 
**Task summary**: Determine the correct Abercrombie classification for a mark/product pair.
 
**Size (samples)**: 100

## Task Description
 
 A particular mark (e.g. a name for a product or service) is only eligible for trademark protection if it is considered to be \textit{distinctive}~\todocite. In assessing whether a mark is distinctive, lawyers and judges follow the framework set out in the case *Abercrombie & Fitch Co. v. Hunting World, Inc.*, which enumerates 5 categories of distinctiveness. These categories characterize the relationship between the dictionary definition of the term used in the mark, and the service or product it is being attached to.
 
 - **Generic**: Generic terms are those which connote the basic nature of articles or services, rather than the more individualized characteristics of a product.
 - **Descriptive**: Descriptive terms identify a characteristic or quality of an article or service, such as color, odor, function, dimensions, or ingredients.
 - **Suggestive**: A suggestive term suggests, rather than describes, some particular characteristic of the goods or services to which it applies. It requires to consumer to exercise the imagination in order to draw a conclusion as to the nature of the goods and services.
 - **Arbitrary**: Arbitrary terms are those that are real words, but arbitrary with respect to the product.
 - **Fanciful**: Fanciful terms are those that are entirely made up, and not found in the English dictionary.
 
 ## Dataset Construction
 
 We manually create a dataset to evaluate a model's ability to classify a mark's distinctiveness (into one of the above 5 categories) with respect to a product. In writing samples, we draw inspiration from similar exercises available in legal casebooks and online resources. We create 20 samples for each category of distinctiveness, and randomly select a single sample from each category to constitute the train set. The remaining 19 samples (for each category) are assigned to the test set (for a total of 95 samples).


## Files

- `train.tsv`: contains samples to be used as in-context demonstrations
- `test.tsv`: contains the evaluation set
- `application_prompt.txt`: contains the prompt that should be used when evaluation application generation
- `ll_rule*_prompt.txt`: contains prompts corresponding to different versions of stating the Abercrombie rule. 
- `rule_stated_prompt.txt`: contains a prompt where the Abercrombie rule is merely stated, and not described.

## Data column names
In `train.tsv` and `test.tsv`, column names correspond to the following:
- `index`: sample identifier
- `answer`: the correct Abercrombie mark. There are five choices: generic, descriptive, suggestive, arbitrary, and fanciful. 
- `text`: a sentence describing the product and the candidate trademark. 
