# SentimentAnalysis_korean_GC   
2020 국어 정보처리 시스템 경진대회 지정분야 출품작입니다.   

# DISCRIPTION   

### Star-N    
기존의 downstream task의 feature가 상대적으로 덜 입력되던 pre-trained model에 downstream의 label에 따른 feature를 더 적용시킬 수 있도록 제작한 model   

### VoSenti   

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

# REFERENCE     
[KoELECTRA](https://github.com/monologg/KoELECTRA)   
[nsmc](https://github.com/e9t/nsmc)
