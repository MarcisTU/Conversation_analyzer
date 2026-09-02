### Podcast speech analysis AI system

- Given long audio file from youtube video it processes different audio deep learning models on this audio including but not limited to (Diarization, ASR, Audio Emotions, Laugh detection)
- Then after obtaining transcript we can summarize into topics, do fact checking etc..

TODO model training:
- for laugh detection actually need to scrape and process data to train the model
- 

## System Architecture

![Current system flow](assets/cur_flow.png)


Setup & running the system:
1) Using provided docker to start the services::
Build the main api container image and start the api
- docker compose build --no-cache api
- docker compose up -d api
Start the broker worker
- docker compose up -d broker_worker
Start the AI services
e.g.::
- docker compose up -d diarization_worker
or use the "workers" profile
- docker compose --profile workers up -d

2) Using separate services scripts to run one by one::
- Before any other AI service model can be used Diarization needs to be processed first.
```
python -m src.services.audio_diarization_service
```
