# Podcast speech analysis AI system

- Given long audio file from youtube video it processes different audio deep learning models on this audio including but not limited to (Diarization, ASR, Audio Emotions, Laugh detection)
- Then after obtaining transcript we can summarize into topics, do fact checking etc..

TODO model training:
- for laugh detection actually need to scrape and process data to train the model
- 

## System Architecture

![Current system flow](assets/cur_flow.png)


### Local development/testing

Setup & running the system:

1. Using provided docker compose and Dockerfile to start the services:

   Build the main api container image and start the api:

   ```bash
   docker compose build --no-cache api
   docker compose up -d api
   ```

   Start the broker worker:

   ```bash
   docker compose up -d broker_worker
   ```

   Start the AI services:

   e.g.:

   ```bash
   docker compose up -d diarization_worker
   ```

   or use the "workers" profile:

   ```bash
   docker compose --profile workers up -d
   ```

2. Using separate services scripts to run one by one:

   * Before any other AI service model can be used Diarization needs to be processed first.

   ```bash
   python -m src.services.audio_diarization_service
   ```

### Production VPS example

I tested creating postgresql database on Supabase.com, then renting an example VPS from https://www.vpsnet.com/lv for hosting fastapi, file storage, broker worker and callback worker services that are launched with docker compose and use internal docker network.

Then when this is setup and tested that it is accesible, i launched external AI diarization service on my local PC and tested file processing which was succesfull.

> ^ For production .env_gc and docker-compose.gc.yml files are used instead.

Running sercies on VPS:

```bash
docker compose --env-file .env_gc -f docker-compose.gc.yml up -d api broker_worker
```

Then on another server/or local PC can start the diarization service (other services as developed)
```bash
docker compose --env-file .env_gc -f docker-compose.gc.yml up -d diarization_worker
```



