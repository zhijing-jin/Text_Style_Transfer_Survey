This is the reading list for text style transfer papers maintained by Zhijing Jin at Max Planck Institute for Intelligent Systems, Tübingen.

A comprehensive overview of the field is introduced in our survey [**Deep Learning for Text Style Transfer: A Survey**](https://arxiv.org/pdf/2011.00416.pdf) (2020) by Di Jin*, Zhijing Jin* (Equal contribution), Zhiting Hu, 
Olga Vechtomova, and Rada Mihalcea.

# Reading List for Text Style Transfer
* [Workshops and Tutorials ](#workshops_and_tutorials)
* [Papers Classified by Method](#papers_classified_by_method)
  * [Unsupervised (Non-parallel Data)](#unsupervised)
    * [Disentanglement](#unsupervised_method_1)
    * [Prototype Editing](#unsupervised_method_2)
    * [Back-Translation / Pseudo Data Construction](#unsupervised_method_3)
    * [Paraphrasing](#unsupervised_method_4)
  * [Semi-Supervised](#semi_supervised)
  * [Supervised](#supervised)
* [Subtasks](#subtasks)
  * [Formality Transfer (Informal <-> Formal)](#formality_transfer)
  * [Politeness Transfer (Impolite -> Polite)](#politeness_transfer)
  * [Simplification (Expert <-> Laymen)](#simplification)
  * [Author/Prose Styles](#author_prose_styles)
  * [Eye-Catchy Rewriting (Plain -> Attractive)](#eye_catchy_rewriting)
  * [Emotion Modification](#emotion_modification)
  * [Detoxification and Debiasing (Toxic/Biased -> Neutral)](#detoxification_and_debiasing)
* [Downstream Applications](#downstream_applications)
  * [Machine Translation with Styles](#machine_translation_with_styles)
  * [Stylized Dialog/Response Generation](#stylized_dialog_response_generation)
  * [Summarization with Styles](#summarization_with_styles)
  * [Simile Generation](#simile_generation)
  * [Story Generation with Styles](#story_generation_with_styles)
  * [Stylized Image Captions](#stylized_image_captions)
  * [Grammatical Error Correction](#grammatical_error_correction)
* [Datasets](#datasets)
* [Evaluation and Analysis](#evaluation_and_analysis)
* [Relevant Fields](#relevant_fields)
  * [Controlled Text Generation (Similar, but not exactly style transfer)](#controlled_text_generation)
  * [Unsupervised Machine Translation](#unsupervised_machine_translation)
  * [Image Style Transfer](#image_style_transfer)
  * [Prototype Editing for Text Generation](#prototype_editing_for_text_generation)
  * [Other Style-Related Papers](#other_style_related_papers)
* [Other Resources](#other_resources)
  * [Review and Thesis](#review_and_thesis)
  * [Other GitHub Repo](#other_github_repo)
* [Copyright](#copyright)


<h2 id="workshops_and_tutorials">Workshops and Tutorials</h2> 

* Stylistic Variation, EMNLP 2017, [[link]](https://sites.google.com/site/workshoponstylisticvariation/)
* Stylistic Variation, NAACL-HLT 2018, [[link]](https://sites.google.com/view/2ndstylisticvariation/home)
* Stylized Text Generation, ACL 2020, [[link]](https://sites.google.com/view/2020-stylized-text-generation/tutorial) [[video-part1]](https://vimeo.com/436479481) [[video-part2]](https://www.youtube.com/watch?v=qSbqVjM-Vik)

<h2 id="papers_classified_by_method">Papers Classified by Method</h2>

<h3 id="unsupervised">Unsupervised (Non-parallel Data)</h3>
<h4 id="unsupervised_method_1">(Unsupervised Method 1) Disentanglement</h4>

1. (2017 ICML) **Sequence to Better Sequence: Continuous Revision of Combinatorial Structures.** _Jonas Mueller, David Gifford, Tommi Jaakkola_. [[paper](http://proceedings.mlr.press/v70/mueller17a/mueller17a.pdf)] [[code](https://bitbucket.org/jwmueller/sequence-to-better-sequence/)]
1. (2017 ICML) **Toward Controlled Generation of Text.** _Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric P. Xing_. [[paper](https://arxiv.org/pdf/1703.00955.pdf)] [[code](https://github.com/asyml/texar/tree/master/examples/text_style_transfer)] [[unofficial code](https://github.com/GBLin5566/toward-controlled-generation-of-text-pytorch)]
1. (2017 NeurIPS) **Style Transfer from Non-Parallel Text by Cross-Alignment.** _Tianxiao Shen, Tao Lei, Regina Barzilay, Tommi Jaakkola_. [[paper](https://papers.nips.cc/paper/7259-style-transfer-from-non-parallel-text-by-cross-alignment.pdf)] [[code](https://github.com/shentianxiao/language-style-transfer)] [[data - Yelp ](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp)]
1. (2017 arXiv) **Zero-Shot Style Transfer in Text Using Recurrent Neural Networks.** _Keith Carlson, Allen Riddell, Daniel Rockmore_. [[paper](https://arxiv.org/pdf/1711.04731v11)] [[code](https://github.com/keithecarlson/Zero-Shot-Style-Transfer)]
1. (2017 NLPCC) **Unsupervised Automatic Text Style Transfer Using LSTM.** __. [[paper](http://tcci.ccf.org.cn/conference/2017/papers/1135.pdf)]
1. (2018 AAAI) **Style Transfer in Text: Exploration and Evaluation.** _Zhenxin Fu, Xiaoye Tan, Nanyun Peng, Dongyan Zhao, Rui Yan_. [[paper](https://arxiv.org/pdf/1711.06861.pdf)] [[code](https://github.com/fuzhenxin/text_style_transfer)]
1. (2018 ICML) **Adversarially Regularized Autoencoders.** _Jake Zhao (Junbo), Yoon Kim, Kelly Zhang, Alexander M. Rush, Yann LeCun_. [[paper](https://arxiv.org/pdf/1706.04223)] [[code](https://github.com/jakezhaojb/ARAE)]
1. (2018 NAACL) **SHAPED: Shared-Private Encoder-Decoder for Text Style Adaptation.** __. [[paper](https://arxiv.org/pdf/1804.04093)]
1. (2018 ACL) **Style Transfer Through Back-Translation.** _Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, Alan W Black_. [[paper](https://arxiv.org/pdf/1804.09000)] [[code](https://github.com/shrimai/Style-Transfer-Through-Back-Translation)]
1. (2018 ACL) **Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach.** _Jingjing Xu, Xu Sun, Qi Zeng, Xuancheng Ren, Xiaodong Zhang, Houfeng Wang, Wenjie Li_. [[paper](https://arxiv.org/pdf/1805.05181)] [[code](https://github.com/lancopku/unpaired-sentiment-translation)]
1. (2018 EMNLP) **Learning Sentiment Memories for Sentiment Modification without Parallel Data.** _Yi Zhang, Jingjing Xu, Pengcheng Yang, Xu Sun_. [[paper](https://arxiv.org/pdf/1808.07311.pdf)] [[code](https://github.com/lancopku/SMAE)]
1. (2018 NeurIPS) **Unsupervised Text Style Transfer using Language Models as Discriminators.** _Zichao Yang, Zhiting Hu, Chris Dyer, Eric P. Xing, Taylor Berg-Kirkpatrick_. [[paper](https://arxiv.org/pdf/1805.11749.pdf)]
1. (2018 NeurIPS) **Content preserving text generation with attribute controls.** _Lajanugen Logeswaran, Honglak Lee, Samy Bengio_. [[paper](https://arxiv.org/pdf/1811.01135.pdf)]
1. (2018 NeurIPS) **Adversarial Text Generation via Feature-Mover's Distance.** _Liqun Chen, Shuyang Dai, Chenyang Tao, Dinghan Shen, Zhe Gan, Haichao Zhang, Yizhe Zhang, Lawrence Carin_. [[paper](https://arxiv.org/pdf/1809.06297)] [[unofficial code](https://github.com/knok/chainer-fm-gan)]
1. (2018 arXiv) **Language Style Transfer from Sentences with Arbitrary Unknown Styles.** _Yanpeng Zhao, Wei Bi, Deng Cai, Xiaojiang Liu, Kewei Tu, Shuming Shi_. [[paper]()]
1. (2018 arXiv) **Structured Content Preservation for Unsupervised Text Style Transfer.** _Youzhi Tian, Zhiting Hu, Zhou Yu_. [[paper](https://arxiv.org/pdf/1810.06526.pdf)] [[code](https://github.com/YouzhiTian/Structured-Content-Preservation-for-Unsupervised-Text-Style-Transfer)]
1. (2019 NAACL) **Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus.** _Hongyu Gong, Suma Bhat, Lingfei Wu, Jinjun Xiong, Wen-mei Hwu_. [[paper](https://arxiv.org/pdf/1903.10671.pdf)]
1. (2019 ICLR) **Multiple-Attribute Text Style Transfer (Rewriting).** _Guillaume Lample, Sandeep Subramanian, Eric Smith, Ludovic Denoyer, Marc'Aurelio Ranzato, Y-Lan Boureau_. [[paper](https://arxiv.org/pdf/1811.00552.pdf)] [[code](https://github.com/facebookresearch/MultipleAttributeTextRewriting)]
1. (2019 ACL) **Disentangled Representation Learning for Non-Parallel Text Style Transfer.** _Vineet John, Lili Mou, Hareesh Bahuleyan, Olga Vechtomova_. [[paper](https://arxiv.org/pdf/1808.04339)] [[code](https://github.com/vineetjohn/linguistic-style-transfer)]
1. (2019 ACL) **Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation.** _Ning Dai, Jianze Liang, Xipeng Qiu, Xuanjing Huang_. [[paper](https://arxiv.org/pdf/1905.05621.pdf)]
1. (2019 ACL) **A Hierarchical Reinforced Sequence Operation Method for Unsupervised Text Style Transfer.** _Chen Wu, Xuancheng Ren, Fuli Luo, Xu Sun_. [[paper](https://www.aclweb.org/anthology/P19-1482.pdf)] [[code](https://github.com/ChenWu98/Point-Then-Operate)]
1. (2019 ACL) **Generating Sentences from Disentangled Syntactic and Semantic Spaces.** _Yu Bao, Hao Zhou, Shujian Huang, Lei Li, Lili Mou, Olga Vechtomova, Xin-yu Dai, Jiajun Chen_. [[paper](https://www.aclweb.org/anthology/P19-1602.pdf)] [[code](https://github.com/baoy-nlp/DSS-VAE)]
1. (2019 IJCAI) **A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer.** _Fuli Luo, Peng Li, Jie Zhou, Pengcheng Yang, Baobao Chang, Zhifang Sui, Xu Sun_. [[paper](https://arxiv.org/pdf/1905.10060.pdf)] [[code](https://github.com/luofuli/DualLanST)]
1. (2019 NeurIPS) **Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation.** _Ke Wang, Hang Hua, Xiaojun Wan_. [[paper](https://arxiv.org/pdf/1905.12926.pdf)] [[code](https://github.com/Nrgeup/controllable-text-attribute-transfer)]
1. (2019 ICAART) **Towards Controlled Transformation of Sentiment in Sentences.** _Wouter Leeftink, Gerasimos Spanakis_. [[paper](https://arxiv.org/pdf/1901.11467.pdf)]
1. (2019 WNGT) **Decomposing Textual Information For Style Transfer.** _Ivan P. Yamshchikov, Viacheslav Shibaev, Aleksander Nagaev, Jürgen Jost, Alexey Tikhonov_. [[paper](https://www.aclweb.org/anthology/D19-5613.pdf)] [[code](https://github.com/VAShibaev/textstyletransfer)]
1. (2019 arXiv) **Grammatical Error Correction and Style Transfer via Zero-shot Monolingual Translation.** _Elizaveta Korotkova, Agnes Luhtaru, Maksym Del, Krista Liin, Daiga Deksne, Mark Fishel_. [[paper](https://arxiv.org/pdf/1903.11283.pdf)]
1. (2019 arXiv) **Zero-Shot Fine-Grained Style Transfer: Leveraging Distributed Continuous Style Representations to Transfer To Unseen Styles.** _Eric Michael Smith, Diana Gonzalez-Rico, Emily Dinan, Y-Lan Boureau_. [[paper]()]
1. (2020 AAAI) **Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning.** _Dayiheng Liu, Jie Fu, Yidan Zhang, Chris Pal, Jiancheng Lv_. [[paper](https://arxiv.org/pdf/1905.12304.pdf)] [[code](https://github.com/dayihengliu/)]
1. (2020 ICML) **On Variational Learning of Controllable Representations for Text without Supervision.** _Peng Xu, Jackie Chi Kit Cheung, Yanshuai Cao_. [[paper](https://arxiv.org/pdf/1905.11975.pdf)] [[code](https://github.com/BorealisAI/CP-VAE)]
1. (2020 ICLR) **A Probabilistic Formulation of Unsupervised Text Style Transfer.** _Junxian He, Xinyi Wang, Graham Neubig, Taylor Berg-Kirkpatrick_. [[paper](https://openreview.net/forum?id=HJlA0C4tPS)] [[code](https://github.com/cindyxinyiwang/deep-latent-sequence-model)]
1. (2020 ACL) **Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer.** _Chulun Zhou, Liangyu Chen, Jiachen Liu, Xinyan Xiao, Jinsong Su, Sheng Guo, Hua Wu_. [[paper](https://arxiv.org/pdf/2005.02049.pdf)] [[code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-WST)]
1. (2020 ACL) **Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders.** _Yu Duan, Canwen Xu, Jiaxin Pei, Jialong Han, Chenliang Li_. [[paper](https://arxiv.org/pdf/1911.03882.pdf)]
1. (2020 IJCAI) **Text Style Transfer via Learning Style Instance Supported Latent Space.** _Xiaoyuan Yi, Zhenghao Liu, Wenhao Li, Maosong Sun_. [[paper](https://www.ijcai.org/Proceedings/2020/0526.pdf)] [[video](https://www.ijcai.org/proceedings/2020/video/24313)] [[code](github.com/XiaoyuanYi/StyIns)]
1. (2020 EMNLP) **Contextual Text Style Transfer.** _Yu Cheng, Zhe Gan, Yizhe Zhang, Oussama Elachqar, Dianqi Li, Jingjing Liu_. [[paper](https://www.aclweb.org/anthology/2020.findings-emnlp.263.pdf)] [[code](https://github.com/ych133/CAST)]
1. (2020 COLING) **Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer.** _Yufang Huang, Wentao Zhu, Deyi Xiong, Yiye Zhang, Changjian Hu, Feiyu Xu_. [[paper](https://arxiv.org/pdf/2010.00735.pdf)]
1. (2020 COLING) **How Positive Are You: Text Style Transfer using Adaptive Style Embedding.** _Heejin Kim, Kyung-Ah Sohn_. [[paper](https://www.aclweb.org/anthology/2020.coling-main.191.pdf)] [[code](https://github.com/kinggodhj/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding)]
1. (2020 ECIR) **SentiInc: Incorporating Sentiment Information into Sentiment Transfer Without Parallel Data.** _Kartikey Pant, Yash Verma, Radhika Mamidi_. [[paper](https://link.springer.com/content/pdf/10.1007%2F978-3-030-45442-5_39.pdf)]
1. (2020 ECIR) **Reinforced Rewards Framework for Text Style Transfer.** _Abhilasha Sancheti, Kundan Krishna, Balaji Vasan Srinivasan, Anandhavelu Natarajan_. [[paper](https://arxiv.org/pdf/2005.05256.pdf)]
1. (2020 arXiv) **ST2: Small-data Text Style Transfer via Multi-task Meta-Learning.** _Xiwen Chen, Kenny Q. Zhu_. [[paper](https://arxiv.org/pdf/2004.11742.pdf)]
1. (2020 arXiv) **Learning to Generate Multiple Style Transfer Outputs for an Input Sentence.** _Kevin Lin, Ming-Yu Liu, Ming-Ting Sun, Jan Kautz_. [[paper](https://arxiv.org/pdf/2002.06525.pdf)]
1. (2021 NAACL) **On Learning Text Style Transfer with Direct Rewards.**
_Yixin Liu, Graham Neubig, John Wieting_. [[paper](https://arxiv.org/pdf/2010.12771.pdf)]
1. (2021 NAACL) **Multi-Style Transfer with Discriminative Feedback on Disjoint Corpus.**
_Navita Goyal, Balaji Vasan Srinivasan, Anandhavelu Natarajan, Abhilasha Sancheti_. [[paper](https://arxiv.org/pdf/2010.11578.pdf)]
1. (2021 ACL) **A Hierarchical VAE for Calibrating Attributes while Generating Text using Normalizing Flow.** _Bidisha Samanta, Mohit Agrawal, NIloy Ganguly_. [[paper](https://aclanthology.org/2021.acl-long.187.pdf)]
1. (2021 ACL) **Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization.**
_Dongkyu Lee, Zhiliang Tian, Lanqing Xue and Nevin L. Zhang_. [[paper](https://arxiv.org/pdf/2108.00449.pdf)]
1. (2021 ACL) **Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer.**
_Sharmila Reddy Nangi, Niyati Chhaya, Sopan Khosla, Nikhil Kaushik and Harshit Nyati_. [[paper](https://aclanthology.org/2021.acl-short.7.pdf)]
1. (2021 ACL) **TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling.**
_Parker Riley, Noah Constant, Mandy Guo, Girish Kumar, David Uthus, Zarana Parekh_. [[paper](https://aclanthology.org/2021.acl-long.293.pdf)]

<h4 id="unsupervised_method_2">(Unsupervised Method 2) Prototype Editing</h4>

1. (2018 TACL) **Generating sentences by editing prototypes.** _Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang_. [[paper](https://arxiv.org/pdf/1709.08878.pdf)]
1. (2018 NAACL) **Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer.** _Juncen Li, Robin Jia, He He, Percy Liang_. [[paper](https://arxiv.org/pdf/1804.06437)] [[code](https://github.com/lijuncen/Sentiment-and-Style-Transfer)]
1. (2019 IJCAI) **Mask and Infill: Applying Masked Language Model to Sentiment Transfer.** _Xing Wu, Tao Zhang, Liangjun Zang, Jizhong Han, Songlin Hu_. [[paper](https://arxiv.org/pdf/1908.08039)] [[code](https://github.com/lancopku/Unpaired-Sentiment-Translation)]
1. (2019 EMNLP) **Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer.** _Akhilesh Sudhakar, Bhargav Upadhyay, Arjun Maheswaran_. [[paper](https://arxiv.org/pdf/1908.09368)] [[code](https://github.com/agaralabs/transformer-drg-style-transfer)]
1. (2019 EMNLP) **Style Transfer for Texts: Retrain, Report Errors, Compare with Rewrites.** _Alexey Tikhonov, Viacheslav Shibaev, Aleksander Nagaev, Aigul Nugmanova, Ivan P. Yamshchikov_. [[paper](https://arxiv.org/pdf/1908.06809.pdf)] [[code](https://github.com/VAShibaev/text_style_transfer)]
1. (2020 EMNLP) **Unsupervised Text Style Transfer with Padded Masked Language Models.** _Eric Malmi, Aliaksei Severyn, Sascha Rothe_. [[paper](https://arxiv.org/pdf/2010.01054.pdf)]
1. (2020 INLG) **Stable Style Transformer: Delete and Generate Approach with Encoder-Decoder for Text Style Transfer.** _Joosung Lee_. [[paper](https://arxiv.org/pdf/2005.12086.pdf)] [[code](https://github.com/rungjoo/Stable-Style-Transformer)]
1. (2021 ACL Findings) **LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer.**
_Machel Reid, Victor Zhong_. [[paper](https://arxiv.org/pdf/2105.08206.pdf)]
1. (2021 ACL Findings) **NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer.**
_Fei Huang, Zikai Chen, Chen Henry Wu, Qihan Guo, Xiaoyan Zhu, Minlie Huang_. [[paper](https://arxiv.org/pdf/2106.02210.pdf)]


<h4 id="unsupervised_method_3">(Unsupervised Method 3) Back-Translation / Pseudo Data Construction</h4>

1. (2018 EMNLP) **QuaSE: Sequence Editing under Quantifiable Guidance.** _Yi Liao, Lidong Bing, Piji Li, Shuming Shi, Wai Lam, Tong Zhang_. [[paper](https://arxiv.org/pdf/1804.07007.pdf)] [[code](https://bitbucket.org/leoeaton/quase/src/master/)]
1. (2018 arXiv) **Style Transfer as Unsupervised Machine Translation.** __. [[paper](https://arxiv.org/pdf/1808.07894.pdf)]
1. (2019 EMNLP) **IMaT: Unsupervised Text Attribute Transfer via Iterative Matching and Translation.** _Zhijing Jin, Di Jin, Jonas Mueller, Nicholas Matthews, Enrico Santus_. [[paper](https://arxiv.org/pdf/1901.11333.pdf)] [[code](https://github.com/zhijing-jin/IMaT)]
1. (2019 RANLP) **Large-scale Hierarchical Alignment for Data-driven Text Rewriting.** _Nikola I. Nikolov, Richard H.R. Hahnloser_. [[paper](https://arxiv.org/pdf/1810.08237.pdf)] [[code](https://github.com/ninikolov/lha)]
1. (2020 NeurIPS) **Unsupervised Text Generation by Learning from Search.** _Jingjing Li, Zichao Li, Lili Mou, Xin Jiang, Michael R. Lyu, Irwin King_. [[paper](https://papers.nips.cc/paper/2020/file/7a677bb4477ae2dd371add568dd19e23-Paper.pdf)]


<h4 id="unsupervised_method_4">(Unsupervised Method 4) Paraphrasing</h4>

1. (2020 EMNLP) **Reformulating Unsupervised Style Transfer as Paraphrase Generation.** _Kalpesh Krishna, John Wieting, Mohit Iyyer_. [[paper](https://arxiv.org/pdf/2010.05700.pdf)] [[video](https://slideslive.com/38938942/reformulating-unsupervised-style-transfer-as-paraphrase-generation)] [[code](https://github.com/martiansideofthemoon/style-transfer-paraphrase)]
1. (2020 EMNLP) **DGST: a Dual-Generator Network for Text Style Transfer.** _Xiao Li, Guanyi Chen, Chenghua Lin, Ruizhe Li_. [[paper](https://arxiv.org/pdf/2010.14557.pdf)]
1. (2020 PETS) **Effective writing style imitation via combinatorial paraphrasing.** _Tommi Gröndahl, N. Asokan_. [[paper](https://arxiv.org/pdf/1905.13464.pdf)]


<h3 id="semi_supervised">Semi-Supervised</h3>

1. (2019 EMNLP) **Semi-supervised Text Style Transfer: Cross Projection in Latent Space.** _Mingyue Shang, Piji Li, Zhenxin Fu, Lidong Bing, Dongyan Zhao, Shuming Shi, Rui Yan_. [[paper](https://arxiv.org/pdf/1909.11493.pdf)]


<h2 id="subtasks">Subtasks</h2>

<h3 id="formality_transfer">Formality Transfer (Informal <-> Formal)</h3>

1. (2018 NAACL) **Dear Sir or Madam, May I introduce the YAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer.** _Sudha Rao, Joel Tetreault_. [[paper](https://arxiv.org/pdf/1803.06535.pdf)] [[code](https://github.com/raosudha89/GYAFC-corpus)] [[data - Formality ](https://github.com/raosudha89/GYAFC-corpus)]
1. (2019 AAAI) **Unsupervised Controllable Text Formalization.** _Parag Jain, Abhijit Mishra, Amar Prakash Azad, Karthik Sankaranarayanan_. [[paper](https://arxiv.org/pdf/1809.04556)] [[code](https://github.com/parajain/uctf)]
1. (2019 arXiv) **Formality Style Transfer with Hybrid Textual Annotations.** _Ruochen Xu, Tao Ge, Furu Wei_. [[paper](https://arxiv.org/pdf/1903.06353.pdf)]
1. (2019 EMNLP) **Domain Adaptive Text Style Transfer.** _Dianqi Li, Yizhe Zhang, Zhe Gan, Yu Cheng, Chris Brockett, Ming-Ting Sun, Bill Dolan_. [[paper](https://arxiv.org/pdf/1908.09395.pdf)] [[code](https://github.com/cookielee77/DAST)]
1. (2019 EMNLP) **Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer.** _Yunli Wang, Yu Wu, Lili Mou, Zhoujun Li, Wenhan Chao_. [[paper](https://www.aclweb.org/anthology/D19-1365/)] [[code](https://github.com/jimth001/formality_emnlp19)]
1. (2020 COLING) **Formality Style Transfer with Shared Latent Space.** _Yunli Wang, Yu Wu, Lili Mou, Zhoujun Li, Wenhan Chao_. [[paper](https://www.aclweb.org/anthology/2020.coling-main.203.pdf)] [[code](https://github.com/jimth001/formality_style_transfer_)]
1. (2020 ACL) **Parallel Data Augmentation for Formality Style Transfer.** _Yi Zhang, Tao Ge, Xu Sun_. [[paper](https://arxiv.org/pdf/2005.07522.pdf)] [[code](https://github.com/lancopku/Augmented_Data_for_FST)]
1. (2021 NAACL) **Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer.** _Eleftheria Briakou, Di Lu, Ke Zhang and Joel Tetreault_. [[paper](https://aclanthology.org/2021.naacl-main.256.pdf)]
1. (2021 ACL) **Improving Formality Style Transfer with Context-Aware Rule Injection.** _Zonghai Yao and Hong Yu_. [[paper](https://arxiv.org/pdf/2106.00210.pdf)]
1. (2021 ACL) **Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer.**
_Huiyuan Lai, Antonio Toral and Malvina Nissim_. [[paper](https://aclanthology.org/2021.acl-short.62.pdf)]
 
<h3 id="politeness_transfer">Politeness Transfer (Impolite -> Polite)</h3>

1. (2018 TACL) **Polite dialogue generation without parallel data.** _Tong Niu, Mohit Bansal_. [[paper](https://watermark.silverchair.com/tacl_a_00027.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAqkwggKlBgkqhkiG9w0BBwagggKWMIICkgIBADCCAosGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMdDYQr39GPklZ_M-TAgEQgIICXFbyhaOxmtgwYYgXOGAJtZfP2w2pVWIg5HzgDDYKnt9hJ2XkY5a-uplf_UPSoyRcAlEoIOYhHnL9i1CzChK37bA-y5vUQdldwBYc5pToXaMl1QCFveKDxANV8WMeAYPbLEVNJnZnpYlRGs9AXYqjlfMeeL7cUAwV0aR5cwq8qwrQt-NLUUHGT3hIu0Z0Dhwx878xafjmR2IBbgtYhsdZcfXAHiEtoZCczspDghywH_gV6Q4FShjMLS1B9Djx8EpbBW15UAam0-k6KgGj8ZsevrnnCzROemaHXdpWhTIzVTnkkvsNeHaj-1ML9x9MjySUhsB6DW0nebkc6mbrQiZyCSdXB2nP429yYUwsjDKB_NjmbjB-bwfmgAoQnbK9_QNfDNCOPmAtsyWCRvCzy1SnaDY4WDa3zB8AeWJtTwoYh0VH8SLOE_7McVwoq0R65cDtikNW0uoZueANTFtIoFyNFXCvBMNMJzAd0txymZp_BPggE3iVejYUR7LCxfdPhnknP9yMzyBnKuX9HoWMhQlbXyl74MV2Oi7vwcPEFUDo2hlYilWNGpJfcfvlUEFATnbeftJ5H96f-L3q5z6zNPWujCY5BaO_MxuTosTB6SusA8h95ly7DC8mAz4IJBgEGrlov89SXROEWcbKYMLrSJdqD9LbDmU6lvc2GFhbppKnwvYN3L6zAcbIUSgAecSU2Ognt3FSDcOtKX255nTcO4EaLDa7iBcfftHLkYq2uCQqA_2jAaw-F6LORy5THOOP-CWPTzHzdVG120Qkqz-2ldoRAfe4Qkcoq58eUx9GlcE)]
1. (2020 ACL) **Politeness Transfer: A Tag and Generate Approach.** _Aman Madaan, Amrith Setlur, Tanmay Parekh, Barnabas Poczos, Graham Neubig, Yiming Yang, Ruslan Salakhutdinov, Alan W Black, Shrimai Prabhumoye_. [[paper](https://arxiv.org/pdf/2004.14257.pdf)] [[video](Politeness https://github.com/tag-and-generate/)] [[code](https://github.com/tag-and-generate/)]

<h3 id="simplification">Simplification (Expert <-> Laymen)</h3>

Wikipedia simplification:

1. (2010 COLING) **A monolingual tree-based translation model for sentence simplification.** _Zhemin Zhu, Delphine Bernhard, and Iryna Gurevych_. [[paper](https://www.aclweb.org/anthology/C10-1152.pdf)]
1. (2021 NAACL) **Controllable Text Simplification with Explicit Paraphrasing.** _Mounica Maddela, Fernando Alva-Manchego, Wei Xu_. [[paper](https://arxiv.org/pdf/2010.11004.pdf)]
 
Medical text simplication:
1. (2019 KDD) **Unsupervised Clinical Language Translation.** _Wei-Hung Weng, Yu-An Chung, Peter Szolovits_. [[paper](https://arxiv.org/pdf/1902.01177.pdf)] Dataset: MIMIC-III (non-parallel) 59K documents.
1. (2019 WWW) **Evaluating neural text simplification in the medical domain.** _Laurens Van den Bercken, Robert-Jan Sips, Christoph Lofi_. [[paper](https://dl.acm.org/doi/pdf/10.1145/3308558.3313630?casa_token=WQIEtRWIgTQAAAAA:nJFh6PMhJ6OewHu9KevGbYQ_nMGFwZG3Tv1FURmDuOsZ7UgvwoVHaolirEzMRuK2PpUjjsU6t8o)] Dataset: 2.2K Expert-to-Layman conversion
1. (2020 ACL) **Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen.** _Yixin Cao, Ruihao Shui, Liangming Pan, Min-Yen Kan, Zhiyuan Liu, Tat-Seng Chua_. [[paper](https://arxiv.org/pdf/2005.00701.pdf)] [[code](https://srhthu.github.io/expertise-style-transfer/)] [[data - MSD (parallel) ](https://srhthu.github.io/expertise-style-transfer/)] Dataset: MSD (parallel) 114K sentences.
1. (2021 NAACL) **Paragraph-level Simplification of Medical Texts.** _Ashwin Devaraj, Iain Marshall, Byron Wallace, Junyi Jessy Li_. [[paper](https://aclanthology.org/2021.naacl-main.395.pdf)]

 <h3 id="author_prose_styles">Author/Prose Styles</h3>

1. (2012 COLING) **Paraphrasing for Style.** _Wei Xu, Alan Ritter, Bill Dolan, Ralph Grishman, Colin Cherry_. [[paper](https://www.aclweb.org/anthology/C12-1177.pdf)]
1. (2017 EMNLP Workshop) **Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models.** _Harsh Jhamtani, Varun Gangal, Eduard Hovy, Eric Nyberg_. [[paper](https://arxiv.org/pdf/1707.01161.pdf)] [[code](https://github.com/harsh19/Shakespearizing-Modern-English)]
1. (2018 Royal Society open science) **Evaluating prose style transfer with the Bible.** _Keith Carlson, Allen Riddell, Daniel Rockmore_. [[paper](https://arxiv.org/pdf/1711.04731.pdf)] [[code](https://github.com/keithecarlson/StyleTransferBibleData)] [[data - Bible ](https://github.com/keithecarlson/StyleTransferBibleData)]
1. (2020 AAAI) **Adapting Language Models for Non-Parallel Author-Stylized Rewriting.** _Bakhtiyar Syed, Gaurav Verma, Balaji Vasan Srinivasan, Anandhavelu Natarajan, Vasudeva Varma_. [[paper](https://arxiv.org/pdf/1909.09962)]


<h3 id="eye_catchy_rewriting">Eye-Catchy Rewriting (Plain -> Attractive)</h3>

1. (2016 EMNLP; plain math problem -> engaging stories) **A Theme-Rewriting Approach for Generating Algebra Word Problems.** _Rik Koncel-Kedziorski, Ioannis Konstas, Luke Zettlemoyer, Hannaneh Hajishirzi_. [[paper](https://arxiv.org/pdf/1610.06210.pdf)] Dataset: Star Wars script with 7300 words, Cartoon scripts with 1370 words
1. (2020 ACL) **Hooks in the Headline: Learning to Generate Headlines with Controlled Styles.** _Di Jin, Zhijing Jin, Joey Tianyi Zhou, Lisa Orii, Peter Szolovits_. [[paper](https://arxiv.org/pdf/2004.01980.pdf)] Dataset: 146K NYT+CNN headlines, 500K humorous sentences, 500K romantic sentences, 500K clickbaity headlines.
1. (2021 AAAI) **The Style-Content Duality of Attractiveness: Learning to Write Eye-Catching Headlines via Disentanglement.** _Mingzhe Li, Xiuying Chen, Min Yang, Shen Gao, Dongyan Zhao, Rui Yan_. [[paper](https://arxiv.org/pdf/2012.07419.pdf)]


<h3 id="emotion_modification">Emotion Modification</h3>

1. (2020 ACL Workshop) **Challenges in Emotion Style Transfer: An Exploration with a Lexical Substitution Pipeline.** _David Helbig, Enrica Troiano, Roman Klinger_. [[paper](https://arxiv.org/pdf/2005.07617.pdf)]
2. (2021 WWW) **Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach.** _Ashish Sharma, Inna W. Lin, Adam S. Miner, David C. Atkins, Tim Althoff_. [[paper](https://arxiv.org/pdf/2101.07714.pdf)]

<h3 id="detoxification_and_debiasing">Detoxification and Debiasing (Toxic/Biased -> Neutral)</h3>

1. (2018 ACL) **Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer.** _Cicero Nogueira dos Santos, Igor Melnyk, Inkit Padhi_. [[paper](https://arxiv.org/pdf/1805.07685.pdf)]
1. (2020 AAAI) **Automatically Neutralizing Subjective Bias in Text.** _Reid Pryzant, Richard Diehl Martinez, Nathan Dass, Sadao Kurohashi, Dan Jurafsky, Diyi Yang_. [[paper](https://arxiv.org/pdf/1911.09709.pdf)] [[code](https://github.com/rpryzant/neutralizing-bias)]
1. (2020 COLING) **Towards A Friendly Online Community: An Unsupervised Style Transfer Framework for Profanity Redaction.** _Minh Tran, Yipeng Zhang, Mohammad Soleymani_. [[paper](https://arxiv.org/pdf/2011.00403.pdf)]
1. (2020 EMNLP) **Unsupervised Controllable Revision for Biased Language Correction.** _Xinyao Ma, Maarten Sap, Hannah Rashkin, Yejin Choi_. [[paper](https://www.aclweb.org/anthology/2020.emnlp-main.602.pdf)]
1. (2021 ACL Workshop) **Methods for Detoxification of Texts for the Russian Language.** _Daryna Dementieva, Daniil Moskovskiy, Varvara Logacheva, David Dale, Olga Kozlova, Nikita Semenov, Alexander Panchenko_. [[paper](https://arxiv.org/pdf/2105.09052.pdf)] [[code](https://github.com/skoltech-nlp/rudetoxifier)]

<h2 id="downstream_applications">Downstream Applications</h2>

<h3 id="machine_translation_with_styles">Machine Translation with Styles</h3>

* (2021 NAACL) **Towards Modeling the Style of Translators in Neural Machine Translation.** _Yue Wang, Cuong Hoang, Marcello Federico_. [[paper](https://aclanthology.org/2021.naacl-main.94.pdf)]
* (2016 NAACL) **Controlling Politeness in Neural Machine Translation via Side Constraints.** _Rico Sennrich, Barry Haddow, Alexandra Birch_. [[paper]](https://www.aclweb.org/anthology/N16-1005.pdf)
* (2017 EMNLP) **A Study of Style in Machine Translation: Controlling the Formality of Machine Translation Output.** _Xing Niu, Marianna Martindale, Marine Carpuat_. [[paper]](https://www.aclweb.org/anthology/D17-1299.pdf)
 

<h3 id="stylized_dialog_response_generation">Stylized Dialog/Response Generation</h3>

* Personalizing dialogue agents: I have a dog, do you have pets too?, ACL 2018, [[paper]](https://doi.org/10.18653/v1/P18-1205)
* Structuring Latent Spaces for Stylized Response Generation, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1909.05361)
* Polite Dialogue Generation Without Parallel Data, TACL, [[paper]](https://arxiv.org/pdf/1805.03162)

<h3 id="summarization_with_styles">Summarization with Styles</h3>

* (2020 ACL) **Hooks in the Headline: Learning to Generate Headlines with Controlled Styles.** _Di Jin, Zhijing Jin, Joey Tianyi Zhou, Lisa Orii, Peter Szolovits_. [[paper]](https://arxiv.org/pdf/2004.01980.pdf)
* (2021 NAACL) **Inference Time Style Control for Summarization.** 
_Shuyang Cao and Lu Wang_. [[paper](https://arxiv.org/pdf/2104.01724.pdf)]

<h3 id="simile_generation">Simile Generation</h3>

* Generating similes effortlessly like a Pro: A Style Transfer Approach for Simile Generation, EMNLP 2020, [[paper]](https://arxiv.org/pdf/2009.08942.pdf)

<h3 id="story_generation_with_styles">Story Generation with Styles</h3>

* (2021 ACL Findings) **Stylized Story Generation with Style-Guided Planning.**
_Xiangzhe Kong, Jialiang Huang, Ziquan Tung, Jian Guan, Minlie Huang_. [[paper](https://arxiv.org/pdf/2105.08625.pdf)] 

<h3 id="stylized_image_captions">Stylized Image Captions</h3>

* StyleNet: Generating attractive visual captions with styles, CVPR 2017, [[paper]](https://doi.org/10.1109/CVPR.2017.108)
* Unsupervised Stylish Image Description Generation via Domain Layer Norm, AAAI 2019, [[paper]](https://arxiv.org/pdf/1809.06214)

<h3 id="grammatical_error_correction">Grammatical Error Correction</h3>

* Grammatical Error Correction and Style Transfer via Zero-shot Monolingual Translation, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1903.11283)

<h2 id="datasets">Datasets</h2>

* (2018 NAACL-HLT) **Dear Sir or Madam, May I introduce the YAFC Corpus: Corpus, Benchmarks and Metrics for Formality Style Transfer.** _Sudha Rao, Joel Tetreault_. [[paper]](https://arxiv.org/pdf/1803.06535)
* (2020 AAAI) **A Dataset for Low-Resource Stylized Sequence-to-Sequence Generation.** _Yu Wu, Yunli Wang, Shujie Liu_. [[paper]](https://www.msra.cn/wp-content/uploads/2020/01/A-Dataset-for-Low-Resource-Stylized-Sequence-to-Sequence-Generation.pdf), [[code]](https://github.com/MarkWuNLP/Data4StylizedS2S)
* (2021 NAACL) **Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer.** _Eleftheria Briakou, Di Lu, Ke Zhang and Joel Tetreault_. [[paper](https://aclanthology.org/2021.naacl-main.256.pdf)]
* (2021 NAACL) **StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer.**
_Yiwei Lyu, Paul Pu Liang, Hai Pham, Eduard Hovy, Barnabás Póczos, Ruslan Salakhutdinov and Louis-Philippe Morency_. [[paper](https://aclanthology.org/2021.naacl-main.171.pdf)]
* (2021 ACL) **Style is NOT a single variable: Case Studies for Cross-Style Language Understanding.** _Dongyeop Kang, Eduard Hovy_. [[paper](https://arxiv.org/pdf/1911.03663.pdf)]
* (2021 VLDB Workshop) **Crowdsourcing of Parallel Corpora:the Case of Style Transfer for Detoxification.** _Daryna Dementieva, Sergey Ustyantsev, David Dale, Olga Kozlova, Nikita Semenov, Alexander Panchenko and Varvara Logacheva_ [[paper](http://ceur-ws.org/Vol-2932/paper2.pdf)] [[repo](https://github.com/skoltech-nlp/parallel_detoxification_dataset)]

<h2 id="evaluation_and_analysis">Evaluation and Analysis</h2>

1. (2019 NAACL) **Evaluating Style Transfer for Text.** _Remi Mir, Bjarke Felbo, Nick Obradovich, Iyad Rahwan_. [[paper](https://arxiv.org/pdf/1904.02295.pdf)] [[code](https://github.com/passeul/style-transfer-model-evaluation)]
1. (2019 INLG) **Rethinking Text Attribute Transfer: A Lexical Analysis.** _Yao Fu, Hao Zhou, Jiaze Chen, Lei Li_. [[paper](https://arxiv.org/pdf/1909.12335)] [[code](https://github.com/FranxYao/pivot_analysis)]
1. (2019 WNGT) **Learning Criteria and Evaluation Metrics for Textual Transfer between Non-Parallel Corpora / Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer.** _Richard Yuanzhe Pang, Kevin Gimpel_. [[paper](https://www.aclweb.org/anthology/D19-5614.pdf)]
1. (2019 arXiv) **The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation.** _Richard Yuanzhe Pang_. [[paper](https://arxiv.org/pdf/1910.03747)]
1. (2018 arXiv) **What is wrong with style transfer for texts?.** _Alexey Tikhonov, Ivan P. Yamshchikov_. [[paper](https://arxiv.org/pdf/1808.04365)]
1. (2020 COLING) **Style versus Content: A distinction without a (learnable) difference?.** _Somayeh Jafaritazehjani, Gwénolé Lecorvé, Damien Lolive, John Kelleher_. [[paper](https://www.aclweb.org/anthology/2020.coling-main.197.pdf)]
1. (2021 AAAI) **Style-transfer and Paraphrase: Looking for a Sensible Semantic Similarity Metric.** _Ivan P. Yamshchikov, Viacheslav Shibaev, Nikolay Khlebnikov, Alexey Tikhonov_. [[paper](https://arxiv.org/pdf/2004.05001.pdf)] [[code](https://github.com/VAShibaev/semantic_similarity_metrics)]
1. (2021 NAACL Workshop) **A Review of Human Evaluation for Style Transfer.** _Eleftheria Briakou, Sweta Agrawal, Ke Zhang, Joel Tetreault, Marine Carpuat_. [[paper](https://arxiv.org/pdf/2106.04747.pdf)]

<h2 id="relevant_fields">Relevant Fields</h2>

<h3 id="controlled_text_generation">Controlled Text Generation (Similar, but not exactly style transfer)</h3>

* Toward Controlled Generation of Text, ICML 2017. [[paper]](https://arxiv.org/pdf/1703.00955.pdf) 
* CTRL: A Conditional Transformer Language Model for Controllable Generation, arXiv 2019. [[paper]](https://arxiv.org/pdf/1909.05858.pdf)
* Defending Against Neural Fake News, NeurIPS 2019. (about conditional generation of neural fake news) [[paper]](https://arxiv.org/pdf/1905.12616.pdf)
* Plug and Play Language Models: A Simple Approach to Controlled Text Generation, ICLR 2020. [[paper]](https://openreview.net/pdf?id=H1edEyBKDS)
* Exploring Controllable Text Generation Techniques, COLING 2020. [[paper]](https://arxiv.org/pdf/2005.01822.pdf)
* Controllable and Diverse Text Generation in E-commerce, WWW 2021. [[paper](https://blender.cs.illinois.edu/paper/www2021.pdf)]
1. (2021 ICLR Oral) **A Distributional Approach to Controlled Text Generation.** _Muhammad Khalifa, Hady Elsahar, Marc Dymetman_. [[paper](https://arxiv.org/pdf/2012.11635.pdf)]


<h3 id="unsupervised_machine_translation">Unsupervised Machine Translation</h3>

* Unsupervised neural machine translation, ICLR 2017. [[paper]](https://arxiv.org/pdf/1710.11041.pdf)


<h3 id="image_style_transfer">Image Style Transfer</h3>

* Image style transfer using convolutional neural networks, CVPR 2016, [[paper]](https://doi.org/10.1109/CVPR.2016.265)
* Image-to-image translation with conditional adversarial networks, CVPR 2017, [[paper]](https://doi.org/10.1109/CVPR.2017.632)
* Style augmentation: Data augmentation via style randomization, CVPR 2019 Workshop, [[paper]](http://openaccess.thecvf.com/content_CVPRW_2019/html/Deep_Vision_Workshop/Jackson_Style_Augmentation_Data_Augmentation_via_Style_Randomization_CVPRW_2019_paper.html)

<h3 id="prototype_editing_for_text_generation">Prototype Editing for Text Generation</h3>

* Retrieve and refine: Improved sequence generation models for dialogue, EMNLP 2018 Workshop, [[paper]](https://doi.org/10.18653/v1/w18-5713)
* Guiding neural machine translation with retrieved translation pieces, NAACL 2018, [[paper]](https://doi.org/10.18653/v1/n18-1120)
* A retrieve-and-edit framework for predicting structured outputs, NIPS 2018, [[paper]](http://papers.nips.cc/paper/8209-a-retrieve-and-edit-framework-for-predicting-structured-outputs)
* Extract and edit: An alternative to back-translation for unsupervised neural machine translation, NAACL 2019, [[paper]](https://doi.org/10.18653/v1/n19-1120)
* Simple and effective retrieve-edit-rerank text generation, ACL 2020, [[paper]](https://www.aclweb.org/anthology/2020.acl-main.228/)
* A retrieve-and-rewrite initialization method for unsupervised machine translation, ACL 2020, [[paper]](https://www.aclweb.org/anthology/2020.acl-main.320/)

<h2 id="other_style_related_papers">Other Style-Related Papers</h2>

* Controlling Linguistic Style Aspects in Neural Language Generation, EMNLP 2017 Workshop, [[paper]](https://arxiv.org/pdf/1707.02633)
* Is writing style predictive of scientific fraud?, EMNLP 2017 Workshop, [[paper]](http://www.aclweb.org/anthology/W17-4905)
* Adversarial Decomposition of Text Representation, Arxiv, [[paper]](https://arxiv.org/pdf/1808.09042)
* Transfer Learning for Style-Specific Text Generation, UNK 2018, [[paper]](https://nips2018creativity.github.io/doc/Transfer%20Learning%20for%20Style-Specific%20Text%20Generation.pdf)
* Generating lyrics with variational autoencoder and multi-modal artist embeddings, Arxiv 2018, [[paper]](https://arxiv.org/pdf/1812.08318)
* Generating Sentences by Editing Prototypes, TACL 2018, [[paper]](https://www.aclweb.org/anthology/Q18-1031/)
* ALTER: Auxiliary Text Rewriting Tool for Natural Language Generation, EMNLP 2019, [[paper]](https://arxiv.org/pdf/1909.06564)
* Stylized Text Generation Using Wasserstein Autoencoders with a Mixture of Gaussian Prior, Arxiv 2019, [[paper]](https://arxiv.org/pdf/1911.03828)
* Complementary Auxiliary Classifiers for Label-Conditional Text Generation, AAAI 2020, [[paper]](http://people.ee.duke.edu/~lcarin/AAAI_LiY_6828.pdf), [[code]](https://github.com/s1155026040/CARA)
* Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer, ACL 2020, [[paper]](https://arxiv.org/pdf/2005.02049.pdf)



<h2 id="other_resources">Other Resources</h2>

<h3 id="review_and_thesis">Review and Thesis</h3>

* (2018 JMLR) **Survey of the State of the Art in Natural LanguageGeneration: Core tasks, applications and evaluation.**
_Albert Gatt, Emiel Krahmer_.
[[paper](https://www.jair.org/index.php/jair/article/view/11173/26378) (Section 5-6 are related to styles in text)]
* (2020 arXiv) **Text Style Transfer: A Review and Experiment Evaluation.**
_Zhiqiang Hu, Roy Ka-Wei Lee, Charu C. Aggarwal, Aston Zhang_.
[[paper]](https://arxiv.org/pdf/2010.12742.pdf)
* (2020) **Controllable Text Generation: Should machines reflect the way humans interact in society?**
_Shrimai Prabhumoye_.
[[paper]](https://www.cs.cmu.edu/~sprabhum/docs/proposal.pdf) [[slides]](https://www.cs.cmu.edu/~sprabhum/docs/Thesis_Proposal.pdf)

<h3 id="other_github_repo">Other GitHub Repo</h3>

* [Style-Transfer-in-Text](https://github.com/fuzhenxin/Style-Transfer-in-Text) by Zhenxin Fu

<h2 id="copyright">Copyright</h2>

By [Zhijing Jin](https://zhijing-jin.com).  

**Welcome to open an issue or make a pull request!**

