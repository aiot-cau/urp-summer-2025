# AI 추천 논문 목록
- **중요도**: ●○○ (참고), ●●○ (권장), ●●● (필수)  
- **난이도**: ●○○ (입문), ●●○ (중급), ●●● (고급)  

## 📷 이미지

### 1. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (AlexNet)
- **저자**: A. Krizhevsky, I. Sutskever, G. Hinton  
- **학회/저널**: NeurIPS 2012  
- **중요도**: ●●● | **난이도**: ●○○  
- **요약**: 대규모 CNN을 GPU 기반으로 학습하여 ImageNet 대회에서 압도적인 성능 향상을 달성한 연구로, 딥러닝 기반 컴퓨터비전 시대를 본격적으로 연 논문. ReLU, Dropout, Data Augmentation 등의 핵심 기법을 대중화함.

### 2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
- **저자**: K. He, X. Zhang, S. Ren, J. Sun  
- **학회/저널**: CVPR 2016  
- **중요도**: ●●● | **난이도**: ●●○  
- **요약**: Skip connection을 도입하여 매우 깊은 신경망 학습을 가능하게 만든 획기적인 구조로, 이후의 많은 CNN 모델의 기반이 됨.

### 3. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT)
- **저자**: A. Dosovitskiy et al.  
- **학회/저널**: ICLR 2021  
- **중요도**: ●●● | **난이도**: ●●●  
- **요약**: 이미지 분류 문제를 Transformer 구조로 풀어낸 최초의 연구로, CNN 중심의 컴퓨터비전 패러다임을 Transformer로 확장함.


## 📝 자연어 처리

### 1. [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
- **저자**: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit et al.
- **학회/저널**: NeurIPS 2017
- **중요도**: ●●● | **난이도**: ●●○
- **요약**: Self-Attention 메커니즘을 도입해 순차적인 구조 없이 병렬 처리를 가능하게 만들었으며, 이후 대부분의 NLP 모델의 기반이 됨.

### 2. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (BERT)
- **저자**: J. Devlin, M. Chang, K. Lee, K. Toutanova
- **학회/저널**: NAACL 2019
- **중요도**: ●●● | **난이도**: ●●○
- **요약**: 양방향 Transformer 구조를 활용해 문맥을 정교하게 파악할 수 있는 사전학습 모델을 제안하며, 다양한 NLP 태스크에서 SOTA 달성.

## ⏳ 시계열

### 1. [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) (wav2vec 2.0)
- **저자**: A. Baevski, H. Zhou, A. Mohamed, M. Auli
- **학회/저널**: NeurIPS 2020
- **중요도**: ●●● | **난이도**: ●●●
- **요약**: 음성 데이터에서 라벨 없이도 representation을 학습할 수 있도록 self-supervised 방식의 음성 인식 모델을 제안함.

### 2. [A Decoder-Only Foundation Model for Time-Series Forecasting](https://arxiv.org/abs/2310.10688) (TimesFM)
- **저자**: J. Wu, M. J. Zhang, Y. Chen, Q. Zhou, Y. Wu
- **학회/저널**: ICLR 2023
- **중요도**: ●●○ | **난이도**: ●●○
- **요약**: 시계열 예측을 위해 decoder-only 구조를 기반으로 한 foundation model을 제안하며, 다양한 시계열 데이터에 범용 적용 가능성을 보여줌.

## 🧬 생성형 모델

### 1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE)
- **저자**: D. P. Kingma, M. Welling
- **학회/저널**: ICLR 2014
- **중요도**: ●●● | **난이도**: ●●○
- **요약**: 변분 추론을 활용한 생성 모델로, 잠재 공간에서의 연속적인 표현 학습과 안정적인 생성을 가능하게 한 기초적이면서도 중요한 연구.

### 2. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GAN)
- **저자**: I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, et al.
- **학회/저널**: NeurIPS 2014
- **중요도**: ●●● | **난이도**: ●●●
- **요약**: Generator와 Discriminator가 경쟁하는 적대적 학습 구조로, 생성 모델의 패러다임을 획기적으로 전환시킨 기념비적 논문.

### 3. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (VQ-VAE)
- **저자**: A. van den Oord, O. Vinyals, K. Kavukcuoglu
- **학회/저널**: NeurIPS 2017
- **중요도**: ●●○ | **난이도**: ●●○
- **요약**: 벡터 양자화를 통해 discrete latent space를 형성하고, 안정적인 생성과 압축 표현을 동시에 달성한 생성 모델.

### 4. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
- **저자**: J. Ho, A. Jain, P. Abbeel
- **학회/저널**: NeurIPS 2020
- **중요도**: ●●● | **난이도**: ●●●
- **요약**: 점진적으로 노이즈를 제거하는 과정을 통해 고품질 이미지를 생성하는 diffusion 기반 모델로, 최근 생성 모델의 핵심 트렌드가 됨.

### 5. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion)
- **저자**: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer
- **학회/저널**: CVPR 2022
- **중요도**: ●●○ | **난이도**: ●●○
- **요약**: 잠재 공간에서의 diffusion을 통해 계산 효율성을 높이면서도 고해상도 이미지 생성이 가능한 모델로 실전 활용도가 높음.

---

### 범례
- **중요도**: ● (채워진 동그라미) ○ (빈 동그라미)
- **난이도**: ● (채워진 동그라미) ○ (빈 동그라미)

---

**참고**: 이 목록은 계속 업데이트됩니다. 추가하고 싶은 논문이 있다면 연구실 GitHub 저장소에 기여해주세요.

**마지막 업데이트**: 2025-05-30
