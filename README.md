# Inpainting 
inpainting

## VAE(AutoEncoder)
입력의 특징 추출 후 생성모델 출력
![image](https://user-images.githubusercontent.com/72767245/99287894-31b70380-287e-11eb-89ce-0d4571ccb3d9.png)


## PConv
Reference: <br>
[PAPER] https://arxiv.org/abs/1804.07723 <br>
[CODE] https://github.com/MathiasGruber/PConv-Keras

### 문화재 이미지 복원 기술
우리의　문화재　복원　기술에　대한　작품은　**총　두가지　효과**를　낼　수　있다．

#### 1. 훼손된　이미지에　대한　이미지　복원　기술
- 새로운　PConv2D 케라스　계층의　모델과 불규칙한　구멍의　image inpainting을　위한　Partial Convolutions인　UNet와의　앙상블
- 오픈　데이터셋인　ImageNet으로　학습을 진행
- sketcher를　사용하여　직접　실시간으로　망가트린　사진을　복원　시킬　수　가　있도록 구현
- 이를　통해，　문화재의　사진을　입력했을　시　실시간으로　사용자와 인터랙션이 가능

#### 2. 흑백　사진을　RGB컬러　모델로　복원　시키는　기술
- 흑백사진의　Lab컬러를　사용하여　복원,　Lab에서　명도를　의미하는　L채널을　사용
- 흑백사진의　명도를　사용하여　흑백사진을　RGB컬러　사진으로　복원


### 결과 값
<div>
  <img src="https://user-images.githubusercontent.com/72767245/99153168-419dde80-26ea-11eb-98fb-9aca373f5b84.jpg" width="20%">
  <img src="https://user-images.githubusercontent.com/72767245/99153169-4367a200-26ea-11eb-8ac1-776ee3d89186.png" width="20%">
</div>
<b>원본 이미지와 복원한 이미지</b>

<p align="center"> <div>
  <img src="https://user-images.githubusercontent.com/72767245/99153170-45316580-26ea-11eb-9144-7a6e95323273.png" width="20%">
  <img src="https://user-images.githubusercontent.com/72767245/99153171-45c9fc00-26ea-11eb-9ae2-adb61f17b056.png" width="20%">
</div> </p>
<b>이미지 실시간 복원 결과 값</b>
