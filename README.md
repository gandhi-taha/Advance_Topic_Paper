# Advanced Topic Paper

This repository is designed by me to make training and fine-tuning medium-sized GPT models as easy and efficient as possible. I'm focusing on practicality rather than extensive documentation. While it is still a work in progress, the current state of the code allows you to replicate the training of a GPT-2 model (124M parameters) on the OpenWebText, Shakespeare and PersonaChat dataset. This can be accomplished on a single RTX3060 12GB node in approximately Twenty-four days as this was done for uni paper I decided to skiped the training part and used Pre-trained model. The code itself is intentionally simple and easy to understand. The 'new.py' file consists of a concise training loop, spanning around 300 lines, while 'model.py' defines the GPT model in approximately 300 lines as well. Additionally, there is an option to initialize the model with pre-trained weights from GPT-2 provided by OpenAI. That's all there is to it one can further add OPEN_AI_API="Key" to access GPT-3.5 turbo with a price of 1k Tokens/$0.006 to $0.0002 range that is not tested as they are on  from /v1/fine-tunes for API endpoints that can be found on their Official website. 

## Course

[Advanced Topics in 
Computational Text and 
Media Science 2023](https://www.uni-trier.de/fileadmin/fb2/LDV/CL/Webseite/M.A._NLP/Natural_Language_Processing__M.Sc.__1-F__Moduluebersicht_und_Studienverlaufsplan.pdf)

## Citation

I give credit to the HuggingFace's code from their participation to NeurIPS 2018 dialog competition [ConvAI2](http://convai.io/) which was state-of-the-art on the automatic metrics for Transfer Learning, in which a the scripts were inspired from with my own addition to it and OPEN_AI boilerplate was used.

```bash
@article{DBLP:journals/corr/abs-1901-08149,
  author    = {Thomas Wolf and
               Victor Sanh and
               Julien Chaumond and
               Clement Delangue},
  title     = {TransferTransfo: {A} Transfer Learning Approach for Neural Network
               Based Conversational Agents},
  journal   = {CoRR},
  volume    = {abs/1901.08149},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.08149},
  archivePrefix = {arXiv},
  eprint    = {1901.08149},
  timestamp = {Sat, 02 Feb 2019 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-08149},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Future work

I may release code for evaluating the models on benchmarks.

I'm  still considering to train the model on my own cluster to dive deep into NLP domain to get my hands dirty.

## Supervised by

[Univ.-Prof. Dr. Achim Rettinger](https://www.uni-trier.de/universitaet/fachbereiche-faecher/fachbereich-ii/faecher/computerlinguistik-und-digital-humanities/computerlinguistik/team/prof-dr-achim-rettinger)

