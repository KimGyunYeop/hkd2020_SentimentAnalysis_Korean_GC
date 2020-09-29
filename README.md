# SentimentAnalysis_korean_GC   
2020 국어 정보처리 시스템 경진대회 지정분야 출품작입니다.   

# DESCRIPTION   

### Star-Label    
기존의 downstream task의 feature가 상대적으로 덜 입력되던 pre-trained model에 downstream의 label에 따른 feature를 더 적용시킬 수 있도록 제작한 model   

### VoSenti   
모델 스스로가 각 단어마다 감성 정보를 직접 학습하여 더해줌으로써 단어의 vector를 감성정보와 위치벡터를 합한 값으로 변환 시켜 vector가 감성정보를 반영하여 학습될 수 있도록 한 model

# USAGE   
config/koelectra-base.json을 통해 모델의 parameter를 조정 가능   
### PYTHON   
#### train & eval    
```
pip install -r requirements.txt   
python3 train.py --result_dir train --model_mode FINAL_MODEL --gpu 0
```   

#### test    
```
pip install -r requirements.txt   
python3 test.py --result_dir train --model_mode FINAL_MODEL --test_file nsmc_ratings_test.txt --gpu 0
```   

### DOCKER(ONLY TRAIN&EVAL)   
--gpus all은 container에서 gpu를 사용하기위한 NVIDIA-DOCKER 명령어 이므로 NVIDIA-DOCKER와 nvidia-container-toolkit필요 [link](https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart)
```
sudo docker build --tag hkd2020 .   
sudo docker run --gpus all --rm -e RESULT_DIR=train -e MODEL_MODE=FINAL_MODEL hkd2020
```   

# RESULT   
### NSMC   
|모델|Dev 결과(Accuracy)|
|:---:|:---:|
|KoELECTRA(base)|0.90210|
|Star_Label_AM|0.90332|
|Star_Label_ANN|0.90362|
|VoSenti_for_Word|0.90430|


### ADDITIONAL DATA   
|모델|Dev 결과(Accuracy)|
|:---:|:---:|
|KoELECTRA(base)|0.90332|
|VoSenti_for_Word|0.90528|
|FINAL_MODEL|0.90612|

[data discription](https://github.com/KimGyunYeop/hkd2020_SentimentAnalysis_Korean_GC/tree/master/data/data)

# REFERENCE     
[KoELECTRA](https://github.com/monologg/KoELECTRA)   
[nsmc](https://github.com/e9t/nsmc)
