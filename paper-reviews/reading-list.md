# AIoT 분야 추천 논문 목록

이 문서는 AIoT Lab 학부 인턴을 위한 추천 논문 목록입니다. 논문들은 주제별로 분류되어 있으며, 각 논문의 중요도와 난이도를 표시했습니다.

**중요도**: ★(기초) ~ ★★★(필수)  
**난이도**: 🔍(입문) ~ 🔍🔍🔍(고급)

## 목차
1. [기초 개념 및 서베이](#1-기초-개념-및-서베이)
2. [시계열 데이터 분석](#2-시계열-데이터-분석)
3. [이상 탐지](#3-이상-탐지)
4. [에지 컴퓨팅 및 모델 경량화](#4-에지-컴퓨팅-및-모델-경량화)
5. [스마트 센서 및 IoT 시스템](#5-스마트-센서-및-iot-시스템)
6. [강화학습 기반 AIoT](#6-강화학습-기반-aiot)
7. [AIoT 응용 사례](#7-aiot-응용-사례)
8. [최신 트렌드](#8-최신-트렌드)

## 1. 기초 개념 및 서베이

### 1.1. AIoT 개요 및 동향

1. **"Artificial Intelligence of Things (AIoT): Vision, Architecture, and Applications"** (2020)  
   저자: S. K. Sharma, X. Wang  
   출처: IEEE Internet of Things Journal  
   중요도: ★★★ | 난이도: 🔍  
   요약: AIoT의 개념, 아키텍처, 주요 기술 및 응용 분야를 포괄적으로 소개하는 서베이 논문

2. **"A Survey on Edge Intelligence"** (2020)  
   저자: Z. Zhou, X. Chen, E. Li, L. Zeng, K. Luo, J. Zhang  
   출처: IEEE Access  
   중요도: ★★ | 난이도: 🔍  
   요약: 에지 인텔리전스의 개념, 기술적 과제, 최신 연구 동향을 소개

3. **"Deep Learning for IoT Big Data and Streaming Analytics: A Survey"** (2018)  
   저자: M. Mohammadi, A. Al-Fuqaha, S. Sorour, M. Guizani  
   출처: IEEE Communications Surveys & Tutorials  
   중요도: ★★ | 난이도: 🔍🔍  
   요약: IoT 데이터 분석을 위한 딥러닝 기술의 적용 방법과 과제 논의

### 1.2. 기초 기술 및 방법론

4. **"Attention Is All You Need"** (2017)  
   저자: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin  
   출처: NeurIPS  
   중요도: ★★★ | 난이도: 🔍🔍  
   요약: Transformer 아키텍처를 제안한 논문으로, 시계열 데이터 처리와 AIoT에도 많은 영향을 미침

5. **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** (2019)  
   저자: M. Tan, Q. V. Le  
   출처: ICML  
   중요도: ★★ | 난이도: 🔍🔍  
   요약: 효율적인 CNN 모델 스케일링 방법론으로, 자원 제약적인 IoT 환경에서 유용

## 2. 시계열 데이터 분석

### 2.1. 시계열 예측

6. **"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"** (2018)  
   저자: D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski  
   출처: International Journal of Forecasting  
   중요도: ★★★ | 난이도: 🔍🔍  
   요약: 확률적 시계열 예측을 위한 RNN 기반 모델 제안

7. **"N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting"** (2020)  
   저자: B. N. Oreshkin, D. Carpov, N. Chapados, Y. Bengio  
   출처: ICLR  
   중요도: ★★ | 난이도: 🔍🔍  
   요약: 해석 가능한 딥러닝 기반 시계열 예측 모델

8. **"Transformer-based Deep Survival Analysis"** (2021)  
   저자: L. Wang, J. Chu, J. H. Malmgren, Y. Bai, E. K. Lee, P. Lio  
   출처: KDD  
   중요도: ★★ | 난이도: 🔍🔍🔍  
   요약: Transformer를 활용한 시계열 데이터 분석 및 생존 분석 방법론

### 2.2. 다변량 시계열 분석

9. **"Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks"** (2018)  
   저자: G. Lai, W.-C. Chang, Y. Yang, H. Liu  
   출처: SIGIR  
   중요도: ★★ | 난이도: 🔍🔍  
   요약: 장단기 시간 패턴을 동시에 모델링하는 LSTNet 제안

10. **"Multivariate Time Series Imputation with Generative Adversarial Networks"** (2018)  
    저자: Y. Luo, X. Cai, Y. Zhang, J. Xu, Y. Xiaojie  
    출처: NeurIPS  
    중요도: ★★ | 난이도: 🔍🔍  
    요약: GAN을 활용한 다변량 시계열 데이터의 결측치 처리 방법

## 3. 이상 탐지

### 3.1. 일반적인 이상 탐지 방법

11. **"A Survey on Deep Learning for Anomaly Detection in IoT Time Series"** (2021)  
    저자: H. Ren, Z. Xu, W. Wang, L. Zhuang, C. Ye, X. Wang, G. Zhou  
    출처: ACM Computing Surveys  
    중요도: ★★★ | 난이도: 🔍  
    요약: IoT 시계열 데이터의 이상 탐지를 위한 딥러닝 기법 서베이

12. **"LSTM-based Encoder-Decoder for Multi-sensor Anom...