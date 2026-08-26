### Podcast speech analysis AI system

- Given long audio file from youtube video it processes different audio deep learning models on this audio including but not limited to (Diarization, ASR, Audio Emotions, Laugh detection)
- Then after obtaining transcript we can summarize into topics, do fact checking etc..

TODO model training:
- for laugh detection actually need to scrape and process data to train the model
- 
# 

Setup & running the system:
1) Using provided docker to start the services::
...

2) Using separate services scripts to run one by one::
- Before any other AI service model can be used Diarization needs to be processed first.
```
python -m src.services.audio_diarization_service
```
