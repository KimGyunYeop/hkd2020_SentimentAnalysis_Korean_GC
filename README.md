# SentimentAnalysis_korean_GC   
2020 국어 정보처리 시스템 경진대회 지정분야 출품작입니다.   

# DISCRIPTION   

### Star-N    
기존의 downstream task의 feature가 상대적으로 덜 입력되던 pre-trained model에 downstream의 label에 따른 feature를 더 적용시킬 수 있도록 제작한 model   

### VoSenti   
모델 스스로가 각 단어마다 감성 정보를 직접 학습하여 더해줌으로써 단어의 vector를 감성정보와 위치벡터를 합한 값으로 변환 시켜 vector가 감성정보를 반영하여 학습될 수 있도록 한 model

# USAGE   
config/koelectra-base.json을 통해 모델의 parameter를 조정 가능   
### PYTHON   
```
pip install -r requirement.txt   
python3 train.py --result_dir VoSenti_for_Word_add_aug --model_mode VoSenti_for_Word --gpu 0
```   

### DOCKER
```
sudo docker build --tag hkd2020 .   
sudo docker run hkd2020
```   

# RESULT   
|모델|batch size|Dev 결과(정확도)|
|:---:|:---:|:---:|
|m1|128|acc1|
|m2|128|acc2|
|m3|128|acc3|

# REFERENCE     
[KoELECTRA](https://github.com/monologg/KoELECTRA)   
[nsmc](https://github.com/e9t/nsmc)
