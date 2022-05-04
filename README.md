# Highly Parallel Autoregressive Entity Linking<br>with Discriminative Correction

## Overview

This repository contains the Pytorch implementation of [[1]](#citation)(https://arxiv.org/abs/2109.03792).

Here the [link](https://mega.nz/folder/l4RhnIxL#_oYvidq2qyDIw1sT-KeMQA) to **pre-processed data** used for this work (i.e., training, validation and test splits of AIDA as well as the KB with the entities) and the **released model**.

## Dependencies

* **python>=3.8**
* **pytorch>=1.7**
* **pytorch_lightning>=1.3**
* **transformers>=4.0**

## Structure
* [src](https://github.com/nicola-decao/efficient-autoregressive-EL/tree/master/src): The source code of the model. In [src/data](https://github.com/nicola-decao/efficient-autoregressive-EL/tree/master/src/data) there is an class of a dataset for Entity Linking. In [src/model](https://github.com/nicola-decao/efficient-autoregressive-EL/tree/master/src/model) there are three classes that implement our EL model. One for the Entity Disambiuation part, one for the (autoregresive) Entity Liking part, and one for the entire model (which also contains the training and validation loops).
* [notebooks](https://github.com/nicola-decao/efficient-autoregressive-EL/tree/master/notebooks): Example code for loading our Entity Linking model, evaluate it on AIDA, and run inference on a test document.

## Usage
Please have a look into the [notebooks](https://github.com/nicola-decao/efficient-autoregressive-EL/tree/master/notebooks) folder to see hot to load our Entity Linking model, evaluate it on AIDA, and run inference on a test document.

Here a minimal example that demonstrate how to use our model:
```python
from src.model.efficient_el import EfficientEL
from IPython.display import Markdown
from src.utils import 

# loading the model on GPU and setting the the threshold to the
# optimal value (based on AIDA validation set)
model = EfficientEL.load_from_checkpoint("../models/model.ckpt").eval().cuda()
model.hparams.threshold = -3.2

# loading the KB with the entities
model.generate_global_trie()

# document which we want to apply EL on
s = """CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY . LONDON 1996-08-30 \
West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset \
by an innings and 39 runs in two days to take over at the head of the county championship ."""

# getting spans from the model and converting the result into Markdown for visualization
Markdown(
    get_markdown(
        [s],
        [[(s[0], s[1], s[2][0][0]) for s in spans] 
         for spans in  model.sample([s])]
    )[0]
)
```
Which will generate:

> CRICKET - [LEICESTERSHIRE](https://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club) TAKE OVER AT TOP AFTER INNINGS VICTORY . [LONDON](https://en.wikipedia.org/wiki/London) 1996-08-30 [West Indian](https://en.wikipedia.org/wiki/West_Indies) all-rounder [Phil Simmons](https://en.wikipedia.org/wiki/Philip_Walton) took four for 38 on Friday as [Leicestershire](https://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club) beat [Somerset](https://en.wikipedia.org/wiki/Somerset_County_Cricket_Club) by an innings and 39 runs in two days to take over at the head of the county championship . 


Please cite [[1](#citation)] in your work when using this library in your experiments.

### Training

To train our model you can run the following comand
```bash
python scripts/train.py --gpus ${NUM_GPUS} --acceleration ddp --batch_size 32
```

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT

## Citation
```
[1] De Cao Nicola, Aziz Wilker, & Titov Ivan. (2021).
Highly parallel autoregressive entity linking with discriminative correction.
Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 7662–7669.
https://doi.org/10.18653/v1/2021.emnlp-main.604
```

BibTeX format:
```
@inproceedings{de-cao-etal-2021-highly,
    title = "Highly Parallel Autoregressive Entity Linking with Discriminative Correction",
    author = "De Cao, Nicola  and
      Aziz, Wilker  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.604",
    doi = "10.18653/v1/2021.emnlp-main.604",
    pages = "7662--7669",
}
```

