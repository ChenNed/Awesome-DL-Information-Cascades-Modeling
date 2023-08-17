# Paper List of Deep Learning-based Information Diffusion Modeling

Contributed by [Xueqin Chen](https://scholar.google.com/citations?user=6F-iHFsAAAAJ&hl=zh-CN) from University of Electronic Science and Technology of China (UESTC) and Leiden University. Thanks for [Xovee](https://github.com/Xovee)'s suggestion for improvements!

Information cascades modeling is accomplished via specific prediction tasks, which are categorized into two levels: Micro-level and Macro-level. (1) At micro-level, local patterns of social influence are studied -- e.g., inferring the action status of a user. The approaches predict the likelihood of a user propagating a particular piece of information, or forecast when the next propagation might occur given a certain information cascade. (2) At macro-level, typical studies include cascade size prediction and outbreak prediction (above a certain threshold), both cascade size prediction and outbreak prediction are aiming to estimate the future size (popularity) of the diffusion cascade.

Highly recommended the following survey papers:

1. **Graph representation learning for popularity prediction problem: a survey.**
   *Tiantian Chen, Jianxiong Guo, Weili Wu.* Discrete Mathematics, Algorithms and Applications 2022. [paper](https://arxiv.org/abs/2203.07632)

1. **A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances.**
  *Fan Zhou, Xovee Xu, Goce Trajcevski, Kunpeng Zhang.*
  CSUR 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3433000)

1. **Survey on Deep Learning Based Popularity Prediction (in Chinese).**
*Qi Cao, Huawei Shen，Jinhua Gao，Xueqi Cheng.* (Macro-level)
Journal of Chinese Information Processing 2021. [paper](http://jcip.cipsc.org.cn/CN/abstract/abstract3082.shtml)

## Micro-level
1. **DyDiff-VAE: A Dynamic Variational Framework for Information Diffusion Prediction.**
*Ruijie Wang, Zijie Huang, Shengzhong Liu, Huajie Shao, Dongxin Liu, Jinyang Li, Tianshi Wang, Dachun Sun, Shuochao Yao, Tarek Abdelzaher.*
SIGIR 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3462934)
1. **Information Diffusion Prediction via Dynamic Graph Neural Networks.**
*Zongmai Cao; Kai Han; Jianfu Zhu.*
CSCWD 2021. [paper](https://ieeexplore.ieee.org/document/9437653/authors#authors)
1. **Neural Information Diffusion Prediction with Topic-Aware Attention Network.**
*Hao Wang, Cheng Yang, Chuan Shi.* CIKM 2021. [paper](https://dl.acm.org/doi/10.1145/3459637.3482374) [code](https://github.com/BUPT-GAMMA/TAN)
3. **Joint Learning of User Representation with Diffusion Sequence and Network Structure.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
TKDE 2020. [paper](https://ieeexplore.ieee.org/document/9094385)
1. **HID: Hierarchical Multiscale Representation Learning for Information Diffusion.**
*Zhou Honglu, Shuyuan Xu, and Zouhui Fu.* 
IJCAI 2020. [paper](https://www.ijcai.org/Proceedings/2020/0468.pdf) [code](https://github.com/hongluzhou/HID)
1. **Cascade-LSTM: Predicting Information Cascades using Deep Neural Networks.**
*Sameera Horawalavithana, John Skvoretz, Adriana Iamnitchi.*
arXiv 2020. [paper](https://arxiv.org/pdf/2004.12373.pdf)
1. **Inf-VAE: A Variational Autoencoder Framework to Integrate Homophily and Influence in Diffusion Prediction.**
*Aravind Sankar, Xinyang Zhang, Adit Krishnan, Jiawei Han.*
WSDM 2020. [paper](https://arxiv.org/pdf/2001.00132.pdf) [code](https://github.com/aravindsankar28/Inf-VAE)
1. **DyHGCN: A Dynamic Heterogeneous Graph Convolutional Network to Learn Users’ Dynamic Preferences for Information Diffusion Prediction.**
*Chunyuan Yuan, Jiacheng Li, Wei Zhou, Yijun Lu, Xiaodan Zhang, and Songlin Hu.*
ECMLPKDD 2020. [paper](https://arxiv.org/pdf/2006.05169.pdf)
1. **Neural diffusion model for microscopic cascade study.**
*Cheng Yang, Maosong Sun, Haoran Liu,Shiyi Han, Zhiyuan Liu, and Huanbo Luan.*
 TKDE 2019. [paper](https://arxiv.org/pdf/1812.08933.pdf)
1. **Understanding Information Diffusion via Heterogeneous Information Network Embeddings.**
*Yuan Su, Xi Zhang, Senzhang Wang, Binxing Fang, Tianle Zhang, Philip S. Yu.*
 DASFAA 2019. [paper](https://link.springer.com/chapter/10.1007/978-3-030-18576-3_30)
1. **COSINE: Community-Preserving Social Network Embedding From Information Diffusion Cascades.**
*Yuan Zhang, Tianshu Lyu, Yan Zhang.*
 AAAI 2019. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16364)
1. **A Recurrent Neural Cascade-based Model for Continuous-Time Diffusion.**
*Sylvain Lamprier.*
 ICML 2019. [paper](http://proceedings.mlr.press/v97/lamprier19a.html) [code](https://github.com/lampriers/recCTIC)
1. **Information Diffusion Prediction with Network Regularized Role-based User Representation Learning.**
*Zhitao Wang, Chengyao Chen, Wenjie Li.*
 TKDD 2019. [paper](https://dl.acm.org/citation.cfm?id=3314106)
1. **Hierarchical Diffusion Attention Network.**
*Zhitao Wang, Wenjie Li.*
 IJCAI 2019. [paper](https://pdfs.semanticscholar.org/a8a7/353a42b90d2f43504783dc81ff28c11a9da5.pdf) [code](https://github.com/zhitao-wang/Hierarchical-Diffusion-Attention-Network)
1. **Predicting Future Participants of Information Propagation Trees.**
*Hsing-Huan Chung, Hen-Hsen Huang, Hsin-Hsi Chen.*
 WI 2019. [paper](https://dl.acm.org/citation.cfm?id=3352540)
1. **Community structure enhanced cascade prediction.**
*Chaochao Liu, Wenjun Wang, Yueheng Sun.*
 Neurocomputing 2019. [paper](https://www.sciencedirect.com/science/article/pii/S0925231219307751)
1. **DeepDiffuse: Predicting the 'Who' and 'When' in Cascades.**
*Sathappan Muthiah, Sathappan Muthiah, Bijaya Adhikari, B. Aditya Prakash, Naren Ramakrishnan.*
 ICDM 2018. [paper](http://people.cs.vt.edu/~badityap/papers/deepdiffuse-icdm18.pdf) [code](https://github.com/raihan2108/deep-diffuse)
1. **A sequential neural information diffusion model with structure attention.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 CIKM 2018. [paper](https://dl.acm.org/doi/10.1145/3269206.3269275) [code](https://github.com/zhitao-wang/Sequential-Neural-Information-Diffusion-Model-with-Structure-Attention)
1. **Attention network for information diffusion prediction.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 WWW 2018. [paper](https://dl.acm.org/citation.cfm?id=3186931)
1. **Inf2vec:Latent representation model for social influence embedding.**
*Shanshan Feng, Gao Cong, Arijit Khan,Xiucheng Li, Yong Liu, and Yeow Meng Chee.*
 ICDE 2018. [paper](https://ieeexplore.ieee.org/document/8509310)
1. **Who will share my image? Predicting the content diffusion path in online social networks.**
*W. Hu, K. K. Singh, F. Xiao, J. Han, C.-N. Chuah, and Y. J. Lee.*
 WSDM 2018. [paper](https://arxiv.org/pdf/1705.09275.pdf)
1. **Predicting Temporal Activation Patterns via Recurrent Neural Networks.**
*Giuseppe Manco, Giuseppe Pirrò, Ettore Ritacco.*
 ISMIS 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-030-01851-1_33)
1. **DeepInf: Social Influence Prediction with Deep Learning.**
*Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, Jie Tang.*
 KDD 2018. [paper](https://arxiv.org/pdf/1807.05560.pdf) [code](https://github.com/xptree/DeepInf)
1. **A Variational Topological Neural Model for Cascade-based Diffusion in Networks.**
*Sylvain Lamprier.*
 arXiv 2018. [paper](https://arxiv.org/pdf/1812.10962.pdf)
1. **Topological recurrent neural network for diffusion prediction.**
*Jia Wang, Vincent W Zheng, ZeminLiu, and Kevin Chen-Chuan Chang.*
 ICDM 2017. [paper](https://arxiv.org/pdf/1711.10162.pdf)  [code](https://github.com/vwz/topolstm)
1. **Cascade dynamics modeling with attention-based recurrent neural network.**
*Yongqing Wang, Huawei Shen, Shenghua Liu, Jinhua Gao, and Xueqi Cheng.*
 IJCAI 2017. [paper](https://www.ijcai.org/proceedings/2017/0416.pdf)  [code](https://github.com/Allen517/cyanrnn_project)

## Macro-level
1. **Information Diffusion Prediction via Exploiting Cascade Relationship Diversity.**
   *Xigang Sun, Jingya Zhou, Zhen Wu, Jie Wang.* CSCWD 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10152625)

2. **Explicit time embedding based cascade attention network for information popularity prediction.**
   *Xigang Sun, Jingya Zhou, Ling Liu, Wenqi Wei.* Information Processing & Management 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457323000158)

3. **Wb-MSF: A Large-scale Multi-source Information Diffusion Dataset for Social Information Diffusion Prediction.**
   *Zhen Wu, Jingya Zhou, Jie Wang, Xigang Sun.* CBD 2022 [paper](https://ieeexplore.ieee.org/document/10024585/)

4. **Deep Popularity Prediction in Multi-Source Cascade with HERI-GCN.**
   *Wu Zhen, Jingya Zhou, Ling Liu, Chaozhuo Li, Fei Gu.* ICDE 2022.  [paper](https://ieeexplore.ieee.org/document/9835455) [code](https://github.com/Les1ie/HERI-GCN)

5. **AECasN: An information cascade predictor by learning the structural representation of the whole cascade network with autoencoder.**
  *Xiaodong Feng, Qihang Zhao, Yunkai Li.* Expert Systems With Applications 2021. [paper](https://www.sciencedirect.com/science/article/pii/S0957417421015694)

6. **Pre-training of Temporal Convolutional Neural Networks for Popularity Prediction.**
  *Qi Cao, Huawei Shen, Yuanhao Liu, Jinhua Gao, Xueqi Cheng.* arXiv 2021. [paper](https://arxiv.org/abs/2108.06220) 

7. **Decoupling Representation and Regressor for Long-Tailed Information Cascade Prediction.**
  *Fan Zhou, Liu Yu, Xovee Xu, Goce Trajcevski.*
  SIGIR 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3463104)

8. **CasSeqGCN: Combining Network Structure and Temporal Sequence to Predict Information Cascades.**
  *Yansong Wang, Xiaomeng Wang, Radosław Michalski, Yijun Ran, Tao Jia.*
  arXiv 2021. [paper](https://arxiv.org/abs/2110.06836) [code](https://github.com/MrYansong/CasSeqGCN)

9. **CasGCN: Predicting future cascade growth based on information diffusion graph.**
  *Zhixuan Xu, Minghui Qian, Xiaowei Huang, Jie Meng.* arXiv 2021. [paper](https://arxiv.org/abs/2009.05152)

10. **CCGL: Contrastive Cascade Graph Learning**
  *Xovee Xu, Fan Zhou, Kunpeng Zhang, and Siyuan Liu.*
  arXiv 2021. [paper](https://arxiv.org/pdf/2107.12576.pdf) [code](https://github.com/Xovee/ccgl)

11. **Prediction of information cascades via content and structure proximity preserved graph level embedding.**
   *Xiaodong Feng, Qihang Zhao, Zhen Liu.* Information Sciences 2021. [paper](https://www.sciencedirect.com/science/article/pii/S0020025520312408)

12. **Fully Exploiting Cascade Graphs for Real-time Forwarding Prediction.**
   *Xiangyun Tang, Dongliang Liao, Weijie Huang, Jin Xu, Liehuang Zhu, Meng Shen.* AAAI 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16137) [code](https://github.com/tangguotxy/TempCas)

13. **A Feature Generalization Framework for Social Media Popularity Prediction.**
   *Kai Wang, Penghui Wang, Xin Chen, Qiushi Huang, Zhendong Mao, Yongdong Zhang.* MM 2020. [paper](https://dl.acm.org/doi/10.1145/3394171.3416294)

14. **Variational Information Diffusion for Probabilistic Cascades Prediction.**
   *Fan Zhou, Xovee Xu, Kunpeng Zhang, Goce Trajcevski, Ting Zhong.*
   INFOCOM 2020. [paper](https://ieeexplore.ieee.org/document/9155349)

15. **A Heterogeneous Dynamical Graph Neural Networks Approach to Quantify Scientific Impact.**
   *Fan Zhou, Xovee Xu, Ce Li, Goce Trajcevski, Ting Zhong and Kunpeng Zhang.*
   arXiv 2020. [paper](https://xovee.cn/archive/paper/arXiv_20_HDGNN_Xovee.pdf) [code](https://github.com/Xovee/hdgnn)

16. **CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction.**
   *Fan Zhou, Xovee Xu, Kunpeng Zhang, Siyuan Liu and Goce Trajcevski.*
   arXiv 2020. [paper]() [code](https://github.com/Xovee/casflow)

17. **Continual Information Cascade Learning.**
   *Fan Zhou, Xin Jing, Xovee Xu, Ting Zhong, Goce Trajcevski, Jin Wu.*
   GLOBECOM 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9322124)

18. **Coupled Graph Neural Networks for Predicting the Popularity of Online Content.**
   *Qi Cao, Huawei Shen, Jinhua Gao, Bingzheng Wei, Xueqi Cheng.*
    WSDM 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3336191.3371834) [code](https://github.com/CaoQi92/CoupledGNN)

19. **Learning Bi-directional Social Influence in Information Cascades using Graph Sequence Attention Networks.**
   *Zhenhua Huang, Zhenyu Wang, Rui Zhang, Yangyang Zhao, Fadong Zheng.* WWW 2020. [paper](https://dl.acm.org/doi/10.1145/3366424.3382677)

20. **NPP: A neural popularity prediction model for social media content.**
   *Guandan Chen, Qingchao Kong, Nan Xu, Wenji Mao.*
    Neurocomputing 2019. [paper](https://www.sciencedirect.com/science/article/pii/S0925231218314942)

21. **Cascade2vec: Learning Dynamic Cascade Representation by Recurrent Graph Neural Networks.**
   *Zhenhua Huang, Zhenyu Wang, Rui Zhang.*
    IEEE Access 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8846015) [code](https://github.com/zhenhuascut/Cascade2vec)

22. **Popularity Prediction on Online Articles with Deep Fusion of Temporal Process and Content Features.**
   *Dongliang Liao, Jin Xu, Gongfu Li, Weijie Huang, Weiqing Liu, Jing Li.* AAAI 2019. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/3786)

23. **Information Diffusion Prediction via Recurrent Cascades Convolution.**
   *Xueqin Chen, Fan Zhou, Kunpeng Zhang, Goce Trajcevski, Ting Zhong, and Fengli Zhang.*
    IEEE ICDE 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8731564) [code](https://github.com/ChenNed/CasCN)

24. **Deep Learning Approach on Information Diffusion in Heterogeneous Networks.**
   *Soheila Molaei, Hadi Zare, Hadi Veisi.*
    KBS 2019. [paper](https://arxiv.org/pdf/1902.08810.pdf)

25. **Prediction of Information Cascades via Content and Structure Integrated Whole Graph Embedding.**
   *Xiaodong Feng, Qiang Zhao, Zhen Liu.*
    BSMDMA 2019. [paper](https://www.comp.hkbu.edu.hk/~xinhuang/BSMDMA2019/3.pdf)

26. **Prediction Model for Non-topological Event Propagation in Social Networks.**
   *Zitu Liu, Rui Wang, Yong Liu.*
    ICPCSEE 2019. [paper](https://link.springer.com/chapter/10.1007/978-981-15-0118-0_19)

27. **Learning sequential features for cascade outbreak prediction.**
   *Chengcheng Gou, Huawei Shen, Pan Du, Dayong Wu, Yue Liu, Xueqi Cheng.*
    Knowledge and Information System 2018. [paper](https://link.springer.com/article/10.1007/s10115-017-1143-0)

28. **User-guided hierarchical attention network for multi-modal social image popularity prediction**
   *Wei Zhang, Wen Wang, Jun Wang, Hongyuan Zha.* WWW 2018. [paper](https://dl.acm.org/doi/pdf/10.1145/3178876.3186026) [code](https://github.com/Autumn945/UHAN)

29. **Factorization Meets Memory Network: Learning to Predict Activity Popularity.**
   *Wen Wang, Wei Zhang, Jun Wang.* DASFAA 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-319-91458-9_31) [code](https://github.com/Autumn945/MOOD)

30. **Predicting the Popularity of Online Content with Knowledge-enhanced Neural Networks.**
   *Hongjian Dou, Wayne Xin Zhao, Yuanpei Zhao, Daxiang Dong, Ji-Rong Wen, Edward Y. Chang.*
    KDD 2018. [paper](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_8.pdf)

31. **CAS2VEC: Network-Agnostic Cascade Prediction in Online Social Networks.**
   *Zekarias T. Kefato, Nasrullah Sheikh, Leila Bahri, Amira Soliman, Alberto Montresor, Sarunas Girdzijauskas.*
    SNAMS 2018. [paper](https://people.kth.se/~sarunasg/Papers/Kefato2018cas2vec.pdf) [code](https://github.com/zekarias-tilahun/cas2vec)

32. **Joint Modeling of Text and Networks for Cascade Prediction.**
   *Cheng Li, Xiaoxiao Guo, Qiaozhu Mei.*
    ICWSM 2018. [paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/viewFile/17804/17070)

33. **Sequential prediction of social media popularity with deep temporal context networks.**
   *Bo Wu, Wen-Huang Cheng, Yongdong Zhang, Qiushi Huang, Jintao Li, Tao Mei.* IJCAI 2017. [paper](https://arxiv.org/pdf/1712.04443.pdf) [code](https://github.com/social-media-prediction/TPIC2017)

34. **DeepHawkes: Bridging the gap between prediction and understanding of information cascades.**
   *Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, and Xueqi Cheng.*
    CIKM 2017. [paper](https://dl.acm.org/doi/10.1145/3132847.3132973) [code](https://github.com/CaoQi92/DeepHawkes)

1. **DeepCas: An end-to-end predictor of information cascades.**
*C. Li, J. Ma, X. Guo, and Q. Mei.*
 WWW 2017. [paper](https://arxiv.org/pdf/1611.05373.pdf) [code](https://github.com/chengli-um/DeepCas)


## Micro + Macro
1. **Full-Scale Information Diffusion Prediction With Reinforced Recurrent Networks.**
*Cheng Yang, Hao Wang, Jian Tang, Chuan Shi, Maosong Sun, Ganqu Cui, Zhiyuan Liu.* TNNLS 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9526884) [code](https://github.com/albertyang33/FOREST)
1. **Multi-scale Information Diffusion Prediction with Reinforced Recurrent Networks.**
*Cheng Yang, Jian Tang, Maosong Sun, Ganqu Cui, Zhiyuan Liu.*
IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0560.pdf) [code](https://github.com/albertyang33/FOREST)
1. **Information Cascades Modeling via Deep Multi-Task Learning.**
*Xueqin Chen,  Kunpeng Zhang, Fan Zhou, Goce Trajcevski, Ting Zhong, and Fengli Zhang.*
 SIGIR 2019. [paper](https://dl.acm.org/citation.cfm?id=3331288)
1. **CRPP: Competing Recurrent Point Process for Modeling Visibility Dynamics in Information Diffusion.**
*Avirup Saha, Bidisha Samanta, Niloy Ganguly.*
 CIKM 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3269206.3271726)  [code](https://github.com/ASCARATHIRA/CRPP)





