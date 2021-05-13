```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("../")
```


```python
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from src.model.efficient_el import EfficientEL
from src.data.dataset_el import DatasetEL
from IPython.display import Markdown
from src.utils import get_markdown
```


```python
parser = ArgumentParser()

parser = Trainer.add_argparse_args(parser)

args, _ = parser.parse_known_args()
args.gpus = 1
args.precision = 16

trainer = Trainer.from_argparse_args(args)
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    Using native 16bit precision.



```python
model = EfficientEL.load_from_checkpoint("../models/model.ckpt").eval()
```


```python
model.hparams.threshold = -3.2
model.hparams.test_with_beam_search = False
model.hparams.test_with_beam_search_no_candidates = False
trainer.test(model, test_dataloaders=model.test_dataloader(), ckpt_path=None)
```

    --------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'ed_macro_f1': 0.9203808307647705,
     'ed_macro_prec': 0.9131189584732056,
     'ed_macro_rec': 0.9390283226966858,
     'ed_micro_f1': 0.9348137378692627,
     'ed_micro_prec': 0.9219427704811096,
     'ed_micro_rec': 0.9480490684509277,
     'macro_f1': 0.8363054394721985,
     'macro_prec': 0.8289670348167419,
     'macro_rec': 0.8539509773254395,
     'micro_f1': 0.8550071120262146,
     'micro_prec': 0.8432350158691406,
     'micro_rec': 0.8671125769615173}
    --------------------------------------------------------------------------------



```python
model.generate_global_trie()
```


```python
s = """CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY . LONDON 1996-08-30 \
West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset \
by an innings and 39 runs in two days to take over at the head of the county championship . Their \
stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed \
in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire . \
After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their \
first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three \
for 83 . Trailing by 213 , Somerset got a solid start to their second innings before Simmons stepped in \
to bundle them out for 174 . Essex , however , look certain to regain their top spot after Nasser Hussain \
and Peter Such gave them a firm grip on their match against Yorkshire at Headingley . Hussain , \
considered surplus to England 's one-day requirements , struck 158 , his first championship century of \
the season , as Essex reached 372 and took a first innings lead of 82 . By the close Yorkshire had turned \
that into a 37-run advantage but off-spinner Such had scuttled their hopes , taking four for 24 in 48 balls 
\and leaving them hanging on 119 for five and praying for rain . At the Oval , Surrey captain Chris Lewis , \
another man dumped by England , continued to silence his critics as he followed his four for 45 on Thursday \
with 80 not out on Friday in the match against Warwickshire . He was well backed by England hopeful Mark \
Butcher who made 70 as Surrey closed on 429 for seven , a lead of 234 . Derbyshire kept up the hunt for \
their first championship title since 1936 by reducing Worcestershire to 133 for five in their second \
innings , still 100 runs away from avoiding an innings defeat . Australian Tom Moody took six for 82 but \
Chris Adams , 123 , and Tim O'Gorman , 109 , took Derbyshire to 471 and a first innings lead of 233 . \
After the frustration of seeing the opening day of their match badly affected by the weather , Kent stepped \
up a gear to dismiss Nottinghamshire for 214 . They were held up by a gritty 84 from Paul Johnson but \
ex-England fast bowler Martin McCague took four for 55 . By stumps Kent had reached 108 for three ."""

Markdown(get_markdown([s], [[(s[0], s[1], s[2][0][0]) for s in spans] for spans in  model.sample([s])])[0])
```




CRICKET - [LEICESTERSHIRE](https://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club) TAKE OVER AT TOP AFTER INNINGS VICTORY . [LONDON](https://en.wikipedia.org/wiki/London) 1996-08-30 [West Indian](https://en.wikipedia.org/wiki/West_Indies) all-rounder [Phil Simmons](https://en.wikipedia.org/wiki/Philip_Walton) took four for 38 on Friday as [Leicestershire](https://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club) beat [Somerset](https://en.wikipedia.org/wiki/Somerset_County_Cricket_Club) by an innings and 39 runs in two days to take over at the head of the county championship . Their stay on top , though , may be short-lived as title rivals [Essex](https://en.wikipedia.org/wiki/Essex_County_Cricket_Club) , [Derbyshire](https://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club) and [Surrey](https://en.wikipedia.org/wiki/Surrey_County_Cricket_Club) all closed in on victory while [Kent](https://en.wikipedia.org/wiki/Kent_County_Cricket_Club) made up for lost time in their rain-affected match against [Nottinghamshire](https://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club) . After bowling [Somerset](https://en.wikipedia.org/wiki/Somerset_County_Cricket_Club) out for 83 on the opening morning at [Grace Road](https://en.wikipedia.org/wiki/Grace_Road) , [Leicestershire](https://en.wikipedia.org/wiki/Leicestershire_County_Cricket_Club) extended their first innings by 94 runs before being bowled out for 296 with [England](https://en.wikipedia.org/wiki/England_cricket_team) discard [Andy Caddick](https://en.wikipedia.org/wiki/Andrew_Caddick) taking three for 83 . Trailing by 213 , [Somerset](https://en.wikipedia.org/wiki/Somerset_County_Cricket_Club) got a solid start to their second innings before [Simmons](https://en.wikipedia.org/wiki/Singapore) stepped in to bundle them out for 174 . [Essex](https://en.wikipedia.org/wiki/Essex_County_Cricket_Club) , however , look certain to regain their top spot after [Nasser Hussain](https://en.wikipedia.org/wiki/Nasser_Hussain) and [Peter Such](https://en.wikipedia.org/wiki/Peter_Thomson_(golfer)) gave them a firm grip on their match against [Yorkshire](https://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club) at [Headingley](https://en.wikipedia.org/wiki/Headingley_Stadium) . [Hussain](https://en.wikipedia.org/wiki/Nasser_Hussain) , considered surplus to [England](https://en.wikipedia.org/wiki/England_cricket_team) 's one-day requirements , struck 158 , his first championship century of the season , as [Essex](https://en.wikipedia.org/wiki/Essex_County_Cricket_Club) reached 372 and took a first innings lead of 82 . By the close [Yorkshire](https://en.wikipedia.org/wiki/Yorkshire_County_Cricket_Club) had turned that into a 37-run advantage but off-spinner [Such](https://en.wikipedia.org/wiki/Mark_Broadhurst) had scuttled their hopes , taking four for 24 in 48 balls 
nd leaving them hanging on 119 for five and praying for rain . At the [Oval](https://en.wikipedia.org/wiki/The_Oval) , [Surrey](https://en.wikipedia.org/wiki/Surrey_County_Cricket_Club) captain [Chris Lewis](https://en.wikipedia.org/wiki/Chris_Lewis_(cricketer)) , another man dumped by [England](https://en.wikipedia.org/wiki/England_cricket_team) , continued to silence his critics as he followed his four for 45 on Thursday with 80 not out on Friday in the match against [Warwickshire](https://en.wikipedia.org/wiki/Warwickshire_County_Cricket_Club) . He was well backed by [England](https://en.wikipedia.org/wiki/England_cricket_team) hopeful [Mark Butcher](https://en.wikipedia.org/wiki/Mark_Butcher) who made 70 as [Surrey](https://en.wikipedia.org/wiki/Surrey_County_Cricket_Club) closed on 429 for seven , a lead of 234 . [Derbyshire](https://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club) kept up the hunt for their first championship title since 1936 by reducing [Worcestershire](https://en.wikipedia.org/wiki/Worcestershire_County_Cricket_Club) to 133 for five in their second innings , still 100 runs away from avoiding an innings defeat . [Australian](https://en.wikipedia.org/wiki/Australia) [Tom Moody](https://en.wikipedia.org/wiki/Tommy_Haas) took six for 82 but [Chris Adams](https://en.wikipedia.org/wiki/Chris_Walker_(squash_player)) , 123 , and Tim O'Gorman , 109 , took [Derbyshire](https://en.wikipedia.org/wiki/Derbyshire_County_Cricket_Club) to 471 and a first innings lead of 233 . After the frustration of seeing the opening day of their match badly affected by the weather , [Kent](https://en.wikipedia.org/wiki/Kent_County_Cricket_Club) stepped up a gear to dismiss [Nottinghamshire](https://en.wikipedia.org/wiki/Nottinghamshire_County_Cricket_Club) for 214 . They were held up by a gritty 84 from [Paul Johnson](https://en.wikipedia.org/wiki/Paul_Johnson_(squash_player)) but ex-England fast bowler [Martin McCague](https://en.wikipedia.org/wiki/Martin_McCague) took four for 55 . By stumps [Kent](https://en.wikipedia.org/wiki/Kent_County_Cricket_Club) had reached 108 for three .


