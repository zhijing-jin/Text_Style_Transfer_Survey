This repo collects the papers for text attribute transfer.

## Survey
[**Deep Learning for Text Attribute Transfer: A Survey**](https://arxiv.org/pdf/2011.00416.pdf) (2020) by Di Jin*, Zhijing Jin* (Equal contribution), Zhiting Hu, 
Olga Vechtomova, and Rada Mihalcea.

## Workshops and Tutorials
- Stylistic Variation, EMNLP 2017, [[link]](https://sites.google.com/site/workshoponstylisticvariation/)
- Stylistic Variation, NAACL-HLT 2018, [[link]](https://sites.google.com/view/2ndstylisticvariation/home)
- Stylized Text Generation, ACL 2020, [[link]](https://sites.google.com/view/2020-stylized-text-generation/tutorial) [[video-part1]](https://vimeo.com/436479481) [[video-part2]](https://www.youtube.com/watch?v=qSbqVjM-Vik)

## Paper List
Most are included in our survey, and we also list them here.

### Method Papers
#### Unsupervised (Non-parallel Data)
**Unsupervised Method 1) Disentanglement**
- Sequence to Better Sequence: Continuous Revision of Combinatorial Structures, ICML 2017, [[paper]](http://proceedings.mlr.press/v70/mueller17a.html), [[code]](https://bitbucket.org/jwmueller/sequence-to-better-sequence/)
- Toward Controlled Generation of Text, ICML 2017, [[paper]](https://arxiv.org/pdf/1703.00955), [[official code]](https://github.com/asyml/texar/tree/master/examples/text_style_transfer), [[unofficial code]](https://github.com/GBLin5566/toward-controlled-generation-of-text-pytorch)
- Style Transfer from Non-Parallel Text by Cross-Alignment, NIPS 2017, [[paper]](https://papers.nips.cc/paper/7259-style-transfer-from-non-parallel-text-by-cross-alignment.pdf), [[code]](https://github.com/shentianxiao/language-style-transfer)
- Adversarially Regularized Autoencoders, ICML 2018, [[paper]](https://arxiv.org/pdf/1706.04223), [[code]](https://github.com/jakezhaojb/ARAE)
- Zero-Shot Style Transfer in Text Using Recurrent Neural Networks, Arxiv 2017, [[paper]](https://arxiv.org/pdf/1711.04731v1), [[code]](https://github.com/keithecarlson/Zero-Shot-Style-Transfer)
- Style Transfer in Text: Exploration and Evaluation, AAAI 2018, [[paper]](https://arxiv.org/pdf/1711.06861), [[code]](https://github.com/fuzhenxin/text_style_transfer)
- SHAPED: Shared-Private Encoder-Decoder for Text Style Adaptation, NAACL 2018, [[paper]](https://arxiv.org/pdf/1804.04093)
- Sentiment Transfer using Seq2Seq Adversarial Autoencoders, project for CSYE7245 Northeastern University, [[paper]](https://arxiv.org/pdf/1804.04003)
- Style Transfer Through Back-Translation, ACL 2018, [[paper]](https://arxiv.org/pdf/1804.09000), [[code]](https://github.com/shrimai/Style-Transfer-Through-Back-Translation)
- Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach, ACL 2018, [[paper]](https://arxiv.org/pdf/1805.05181), [[code]](https://github.com/lancopku/unpaired-sentiment-translation)
- Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer, ACL 2018, [[paper]](https://arxiv.org/pdf/1805.07685)
- Unsupervised Text Style Transfer using Language Models as Discriminators, NIPS 2018, [[paper]](https://arxiv.org/pdf/1805.11749)
- Disentangled Representation Learning for Non-Parallel Text Style Transfer, ACL 2019, [[paper]](https://arxiv.org/pdf/1808.04339), [[code]](https://github.com/vineetjohn/linguistic-style-transfer)
- Language Style Transfer from Sentences with Arbitrary Unknown Styles, Arxiv, [[paper]](https://arxiv.org/pdf/1808.04071)
- Learning Sentiment Memories for Sentiment Modification without Parallel Data, EMNLP 2018, [[paper]](https://arxiv.org/pdf/1808.07311), [[code]](https://github.com/lancopku/SMAE)
- Structured Content Preservation for Unsupervised Text Style Transfer, OpenReview 2018, [[paper]](https://openreview.net/forum?id=S1lCbhAqKX)
- Content preserving text generation with attribute controls, NIPS 2018, [[paper]](https://arxiv.org/pdf/1811.01135)
- QuaSE: Sequence Editing under Quantifiable Guidance, EMNLP 2018, [[paper]](http://aclweb.org/anthology/D18-1420)
- Adversarial Text Generation via Feature-Mover's Distance, NeurIPS 2018, [[paper]](https://arxiv.org/pdf/1809.06297), [[unofficial code]](https://github.com/knok/chainer-fm-gan)
- Towards Controlled Transformation of Sentiment in Sentences, ICAART 2019, [[paper]](https://arxiv.org/pdf/1901.11467)
- Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus, NAACL 2019 2019, [[paper]](https://arxiv.org/pdf/1903.10671)
- Grammatical Error Correction and Style Transfer via Zero-shot Monolingual Translation, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1903.11283)
- Multiple-Attribute Text Style Transfer (Rewriting), ICLR 2019, [[paper]](https://openreview.net/forum?id=H1g2NhC5KQ)
- Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation, ACL 2019, [[paper]](https://arxiv.org/pdf/1905.05621)
- A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer, IJCAI 2019, [[paper]](https://arxiv.org/pdf/1905.10060), [[code]](https://github.com/luofuli/DualLanST)
- On Variational Learning of Controllable Representations for Text without Supervision, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1905.11975)
- Revision in Continuous Space: Fine-Grained Control of Text Style Transfer, AAAI 2020, [[paper]](https://arxiv.org/pdf/1905.12304)
- Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation, NIPS 2019, [[paper]](https://arxiv.org/pdf/1905.12926), [[code]](https://github.com/nrgeup/controllable-text-attribute-transfer)
- Disentangled Representation Learning for Non-Parallel Text Style Transfer, ACL 2019, [[paper]](https://www.aclweb.org/anthology/P19-1041), [[code]](https://github.com/vineetjohn/linguistic-style-transfer)
- A Hierarchical Reinforced Sequence Operation Method for Unsupervised Text Style Transfer, ACL 2019, [[paper]](https://www.aclweb.org/anthology/P19-1482), [[code]](https://github.com/ChenWu98/Point-Then-Operate)
- Decomposing Textual Information For Style Transfer, WNGT 2019, [[paper]](https://arxiv.org/pdf/1909.12928)
- Zero-Shot Fine-Grained Style Transfer: Leveraging Distributed Continuous Style Representations to Transfer To Unseen Styles, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1911.03914)
- A Probabilistic Formulation of Unsupervised Text Style Transfer, ICLR 2020, [[paper]](https://openreview.net/forum?id=HJlA0C4tPS), [[code]](https://github.com/cindyxinyiwang/deep-latent-sequence-model)
- Generating sentences from disentangled syntactic and semantic spaces, ACL 2019, [[paper]](https://www.aclweb.org/anthology/P19-1602/), [[code]](https://github.com/baoy-nlp/DSS-VAE)
- SentiInc: Incorporating Sentiment Information into Sentiment Transfer Without Parallel Data, ECIR 2020, [[paper]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-45442-5_39.pdf)
- Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer, COLING 2020, [[paper]](https://arxiv.org/pdf/2010.00735.pdf)

**Unsupervised Method 2) Prototype Editing**
- Generating sentences by editing prototypes, TACL 2018, [[paper]](https://transacl.org/ojs/index.php/tacl/article/view/1296)
- Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer, NAACL 2018, [[paper]](https://arxiv.org/pdf/1804.06437), [[code]](https://worksheets.codalab.org/worksheets/0xe3eb416773ed4883bb737662b31b4948/)
- Mask and Infill: Applying Masked Language Model to Sentiment Transfer, IJCAI 2019, [[paper]](https://arxiv.org/pdf/1908.08039)
- Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1908.09368), [[code]](https://github.com/agaralabs/transformer-drg-style-transfer)
- Style Transfer for Texts: Retrain, Report Errors, Compare with Rewrites, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1908.06809.pdf), [[code]](https://github.com/VAShibaev/text_style_transfer)
- Stable Style Transformer: Delete and Generate Approach with Encoder-Decoder for Text Style Transfer, Arxiv 2020, [[paper]](https://arxiv.org/pdf/2005.12086.pdf)
- Challenges in Emotion Style Transfer: An Exploration with a Lexical Substitution Pipeline, SocialNLP, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.07617.pdf)


**Unsupervised Method 3) Back-Translation / Pseudo Data Construction**
- Incorporating Pseudo-Parallel Data for Quantifiable Sequence Editing, EMNLP 2018, [[paper]](https://arxiv.org/pdf/1804.07007)
- Style Transfer as Unsupervised Machine Translation, Arxiv, [[paper]](https://arxiv.org/pdf/1808.07894)
- IMaT: Unsupervised Text Attribute Transfer via Iterative Matching and Translation, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1901.11333)
- Unsupervised Text Generation by Learning from Search, NeurIPS 2020, [[paper]](https://papers.nips.cc/paper/2020/file/7a677bb4477ae2dd371add568dd19e23-Paper.pdf)

**Unsupervised Method 4) Others**
- Style Transfer Through Multilingual and Feedback-Based Back-Translation, Arxiv 2018, [[paper]](https://arxiv.org/pdf/1809.06284)
- Unsupervised Controllable Text Formalization, AAAI 2019, [[paper]](https://arxiv.org/pdf/1809.04556), [[code]](https://github.com/parajain/uctf)
- Large-scale Hierarchical Alignment for Data-driven Text Rewriting, RANLP 2019, [[paper]](https://arxiv.org/pdf/1810.08237)
- Learning Criteria and Evaluation Metrics for Textual Transfer between Non-Parallel Corpora, Arxiv 2018, [[paper]](https://arxiv.org/pdf/1810.11878)
- Formality Style Transfer with Hybrid Textual Annotations, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1903.06353)
- Domain Adaptive Text Style Transfer, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1908.09395), [[code]](https://github.com/cookielee77/DAST)
- Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.00701.pdf)
- Contextual Text Style Transfer, Arxiv 2020, [[paper]](https://arxiv.org/pdf/2005.00136.pdf)
- Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.02049.pdf)
- ST$^2$: Small-data Text Style Transfer via Multi-task Meta-Learning, Arxiv 2020, [[paper]](https://arxiv.org/pdf/2004.11742)
- Reinforced Rewards Framework for Text Style Transfer, ECIR 2020, [[paper]](https://arxiv.org/pdf/2005.05256)
- Unsupervised Automatic Text Style Transfer Using LSTM, NLPCC 2017, [[paper]](http://tcci.ccf.org.cn/conference/2017/papers/1135.pdf)
- Text Style Transfer via Learning Style Instance Supported Latent Space, IJCAI 2020, [[paper]](https://www.ijcai.org/Proceedings/2020/0526.pdf)
- Learning to Generate Multiple Style Transfer Outputs for an Input Sentence, Arxiv 2020, [[paper]](https://arxiv.org/pdf/2002.06525)
- Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders, ACL 2020, [[paper]](https://arxiv.org/pdf/1911.03882.pdf)
- Unsupervised Text Style Transfer with Padded Masked Language Models, EMNLP 2020, [[paper]](https://arxiv.org/pdf/2010.01054.pdf)
- Reformulating Unsupervised Style Transfer as Paraphrase Generation, EMNLP 2020, [[paper]](https://arxiv.org/pdf/2010.05700.pdf)
- DGST: a Dual-Generator Network for Text Style Transfer, EMNLP 2020, [[paper]](https://arxiv.org/pdf/2010.14557.pdf)
- How Positive Are You: Text Style Transfer Using Adaptive Style Embedding, EMNLP 2020, [[paper]](https://www.aclweb.org/anthology/2020.coling-main.191.pdf)
- Formality Style Transfer with Shared Latent Space, COLING 2020, [[paper]](https://www.aclweb.org/anthology/2020.coling-main.203.pdf)
- Effective writing style transfer via combinatorial paraphrasing, Proc. Priv. Enhancing Technol. 2020, [[paper]](https://doi.org/10.2478/popets-2020-0068)

#### Semi-supervised
- Semi-supervised Text Style Transfer: Cross Projection in Latent Space, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1909.11493)
- Parallel Data Augmentation for Formality Style Transfer, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.07522.pdf)

#### Supervised
- Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models, EMNLP 2017 Workshop, [[paper]](https://arxiv.org/pdf/1707.01161)[[code]](https://github.com/harsh19/Shakespearizing-Modern-English)
- Evaluating prose style transfer with the Bible 2018, [[paper]](https://arxiv.org/pdf/1711.04731)
- Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer, EMNLP 2019, [[paper]](https://www.aclweb.org/anthology/D19-1365/), [[code]](https://github.com/jimth001/formality_emnlp19)
- Automatically Neutralizing Subjective Bias in Text, AAAI 2020, [[paper]](https://nlp.stanford.edu/pubs/pryzant2020bias.pdf)


### Subtasks
**Formality Transfer (Informal <-> Formal)**
- Dear Sir or Madam, May I introduce the YAFC Corpus: Corpus, Benchmarks and Metrics for Formality Style Transfer, NAACL-HLT 2018, [[paper]](https://arxiv.org/pdf/1803.06535)
- Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer, EMNLP 2019, [[paper]](https://www.aclweb.org/anthology/D19-1365/), [[code]](https://github.com/jimth001/formality_emnlp19)
- Parallel Data Augmentation for Formality Style Transfer, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.07522.pdf)

**Politeness Transfer (Impolite -> Polite)**
- Polite dialogue generation without parallel data, TACL 2018, [[paper]](https://doi.org/10.1162/tacl_a_00027)
- Politeness Transfer: A Tag and Generate Approach, ACL 2020, [[paper]](https://arxiv.org/pdf/2004.14257.pdf)

**Simplification (Expert <-> Laymen)**
- Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.00701.pdf)

**Author/Prose Styles**
- Paraphrasing for Style, COLING 2012, [[paper]](https://www.aclweb.org/anthology/C12-1177.pdf)
- Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models, EMNLP 2017 Workshop, [[paper]](https://arxiv.org/pdf/1707.01161)[[code]](https://github.com/harsh19/Shakespearizing-Modern-English)
- Evaluating prose style transfer with the Bible, arXiv 2018, [[paper]](https://arxiv.org/pdf/1711.04731)
- Adapting Language Models for Non-Parallel Author-Stylized Rewriting, AAAI 2020 [[paper]](https://arxiv.org/pdf/1909.09962)

**Emotion Modification**
- Challenges in Emotion Style Transfer: An Exploration with a Lexical Substitution Pipeline, SocialNLP, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.07617.pdf)

**Detoxification and Debiasing (Toxic/Biased -> Neutral)**
- Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer, ACL 2018, [[paper]](https://arxiv.org/pdf/1805.07685)
- Towards A Friendly Online Community: An Unsupervised Style Transfer Framework for Profanity Redaction, COLING 2020, [[paper]](https://arxiv.org/pdf/2011.00403.pdf)
- Automatically Neutralizing Subjective Bias in Text, AAAI 2020, [[paper]](https://nlp.stanford.edu/pubs/pryzant2020bias.pdf)
- PowerTransformer: Unsupervised Controllable Revision for Biased Language Correction, EMNLP 2020, [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.602.pdf)

### Downstream Applications
**Machine Translation with Styles**
- Controlling Politeness in Neural Machine Translation via Side Constraints, NAACL 2016, [[paper]](https://www.aclweb.org/anthology/N16-1005.pdf)
- A Study of Style in Machine Translation: Controlling the Formality of Machine Translation Output, EMNLP 2017, [[paper]](https://www.aclweb.org/anthology/D17-1299.pdf)

**Stylized Dialog/Response Generation**
- Personalizing dialogue agents: I have a dog, do you have pets too?, ACL 2018, [[paper]](https://doi.org/10.18653/v1/P18-1205)
- Structuring Latent Spaces for Stylized Response Generation, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1909.05361)
- Polite Dialogue Generation Without Parallel Data, TACL, [[paper]](https://arxiv.org/pdf/1805.03162)

**Summarization with Styles**
- Hooks in the Headline: Learning to Generate Headlines with Controlled Styles, ACL 2020, [[paper]](https://arxiv.org/pdf/2004.01980.pdf)



**Simile Generation**
- Generating similes effortlessly like a Pro: A Style Transfer Approach for Simile Generation, EMNLP 2020, [[paper]](https://arxiv.org/pdf/2009.08942.pdf)

**Stylized Image Captions**
- Stylenet: Generating attractive visual captions with styles, CVPR 2017, [[paper]](https://doi.org/10.1109/CVPR.2017.108)
- Unsupervised Stylish Image Description Generation via Domain Layer Norm, AAAI 2019, [[paper]](https://arxiv.org/pdf/1809.06214)

**Grammatical Error Correction**
- Grammatical Error Correction and Style Transfer via Zero-shot Monolingual Translation, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1903.11283)

### Datasets
- Dear Sir or Madam, May I introduce the YAFC Corpus: Corpus, Benchmarks and Metrics for Formality Style Transfer, NAACL-HLT 2018, [[paper]](https://arxiv.org/pdf/1803.06535)
- A Dataset for Low-Resource Stylized Sequence-to-Sequence Generation, AAAI 2020, [[paper]](https://www.msra.cn/wp-content/uploads/2020/01/A-Dataset-for-Low-Resource-Stylized-Sequence-to-Sequence-Generation.pdf), [[code]](https://github.com/MarkWuNLP/Data4StylizedS2S)



### Evaluation and Analysis
- Evaluating Style Transfer for Text, NAACL 2019, [[paper1]](https://arxiv.org/pdf/1904.02295), [[paper2]](https://dspace.mit.edu/bitstream/handle/1721.1/119569/1076275047-MIT.pdf?sequence=1)
- Rethinking Text Attribute Transfer: A Lexical Analysis, INLG 2019, [[paper]](https://arxiv.org/pdf/1909.12335), [[code]](https://github.com/FranxYao/pivot_analysis)
- Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer, EMNLP Workshop on Neural Generation and Translation (WNGT) 2019, [[paper]](https://arxiv.org/pdf/1810.11878)
- The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation, WNGT 2019, [[paper]](https://arxiv.org/pdf/1910.03747)
- Style-transfer and Paraphrase: Looking for a Sensible Semantic Similarity Metric, Arxiv 2020, [[paper]](https://arxiv.org/pdf/2004.05001.pdf)
- What is wrong with style transfer for texts? Arxiv, [[paper]](https://arxiv.org/pdf/1808.04365)
- Style versus Content: A distinction without a (learnable) difference?, COLING 2020,	[[paper]](https://www.aclweb.org/anthology/2020.coling-main.197.pdf)


### Relevant Fields
**Controlled Text Generation (Similar, but not exactly style transfer)**
- Toward Controlled Generation of Text, ICML 2017. [[paper]](https://arxiv.org/pdf/1703.00955.pdf) 
- CTRL: A Conditional Transformer Language Model for Controllable Generation, arXiv 2019. [[paper]](https://arxiv.org/pdf/1909.05858.pdf)
- Defending Against Neural Fake News, NeurIPS 2019. (about conditional generation of neural fake news) [[paper]](https://arxiv.org/pdf/1905.12616.pdf)
- Plug and Play Language Models: A Simple Approach to Controlled Text Generation, ICLR 2020. [[paper]](https://openreview.net/pdf?id=H1edEyBKDS)
- Exploring Controllable Text Generation Techniques, COLING 2020, [[paper]](https://arxiv.org/pdf/2005.01822.pdf)

**Unsupervised machine translation**
- Unsupervised neural machine translation, ICLR 2017. [[paper]](https://arxiv.org/pdf/1710.11041.pdf)


**Image style transfer**
- Image style transfer using convolutional neural networks, CVPR 2016, [[paper]](https://doi.org/10.1109/CVPR.2016.265)
- Image-to-image translation with conditional adversarial networks, CVPR 2017, [[paper]](https://doi.org/10.1109/CVPR.2017.632)
- Style augmentation: Data augmentation via style randomization, CVPR 2019 Workshop, [[paper]](http://openaccess.thecvf.com/content_CVPRW_2019/html/Deep_Vision_Workshop/Jackson_Style_Augmentation_Data_Augmentation_via_Style_Randomization_CVPRW_2019_paper.html)

**Prototype Editing for Text Generation**
- Retrieve and refine: Improved sequence generation models for dialogue, EMNLP 2018 Workshop, [[paper]](https://doi.org/10.18653/v1/w18-5713)
- Guiding neural machine translation with retrieved translation pieces, NAACL 2018, [[paper]](https://doi.org/10.18653/v1/n18-1120)
- A retrieve-and-edit framework for predicting structured outputs, NIPS 2018, [[paper]](http://papers.nips.cc/paper/8209-a-retrieve-and-edit-framework-for-predicting-structured-outputs)
- Extract and edit: An alternative to back-translation for unsupervised neural machine translation, NAACL 2019, [[paper]](https://doi.org/10.18653/v1/n19-1120)
- Simple and effective retrieve-edit-rerank text generation, ACL 2020, [[paper]](https://www.aclweb.org/anthology/2020.acl-main.228/)
- A retrieve-and-rewrite initialization method for unsupervised machine translation, ACL 2020, [[paper]](https://www.aclweb.org/anthology/2020.acl-main.320/)


### Other Style-Related Papers
- Controlling Linguistic Style Aspects in Neural Language Generation, EMNLP 2017 Workshop, [[paper]](https://arxiv.org/pdf/1707.02633)
- Is writing style predictive of scientific fraud?, EMNLP 2017 Workshop, [[paper]](http://www.aclweb.org/anthology/W17-4905)
- Adversarial Decomposition of Text Representation, Arxiv, [[paper]](https://arxiv.org/pdf/1808.09042)
- Transfer Learning for Style-Specific Text Generation, UNK 2018, [[paper]](https://nips2018creativity.github.io/doc/Transfer%20Learning%20for%20Style-Specific%20Text%20Generation.pdf)
- Generating lyrics with variational autoencoder and multi-modal artist embeddings, Arxiv 2018, [[paper]](https://arxiv.org/pdf/1812.08318)
- Generating Sentences by Editing Prototypes, TACL 2018, [[paper]](https://www.aclweb.org/anthology/Q18-1031/)
- ALTER: Auxiliary Text Rewriting Tool for Natural Language Generation, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1909.06564)
- Stylized Text Generation Using Wasserstein Autoencoders with a Mixture of Gaussian Prior, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1911.03828)
- Complementary Auxiliary Classifiers for Label-Conditional Text Generation, AAAI 2020, [[paper]](http://people.ee.duke.edu/~lcarin/AAAI_LiY_6828.pdf), [[code]](https://github.com/s1155026040/CARA)
- Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.02049.pdf)




### Other Resources
**Review and Thesis**
- Deep Learning for Text Attribute Transfer: A Survey, arXiv 2020, [[paper]](https://arxiv.org/pdf/2011.00416.pdf)
- Text Style Transfer: A Review and Experiment Evaluation, arXiv 2020, [[paper]](https://arxiv.org/pdf/2010.12742.pdf)
- Controllable Text Generation: Should machines reflect the way humans interact in society?, PhD thesis at CMU 2020, [[paper]](https://www.cs.cmu.edu/~sprabhum/docs/proposal.pdf) [[slides]](https://www.cs.cmu.edu/~sprabhum/docs/Thesis_Proposal.pdf)

**Other GitHub Repo**
- [Style-Transfer-in-Text](https://github.com/fuzhenxin/Style-Transfer-in-Text) by Zhenxin Fu

# Copyright 
By [Zhijing Jin](https://zhijing-jin.com).  

**Welcome to open an issue or make a pull request!**

