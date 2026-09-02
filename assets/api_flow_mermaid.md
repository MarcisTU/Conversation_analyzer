sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant DB as PostgreSQL
    participant R as RabbitMQ
    participant B as Broker Worker
    participant W as AI Worker
    participant CB as Callback

    C->>API: Submit audio + features
    API->>DB: Create Task + FeaturesInTask
    API->>R: TaskPayload
    R->>B: New task

    B->>DB: Load ordered features
    B->>DB: Find next WAITING feature

    alt Worker available
        B->>R: Publish feature job
        R->>W: Worker-specific job
        B->>DB: Feature → PROCESSING

        W->>W: Download audio from MinIO
        W->>W: Run inference
        W->>R: WorkerResponsePayload
        R->>B: Worker response

        B->>DB: Feature → READY + result
        B->>DB: Load next feature

        alt More features
            B->>R: Dispatch next feature
            R->>W: Next feature job
        else All features complete
            B->>DB: Task → READY
            B->>DB: Save TaskResultsFinal
            B->>R: Publish callback
            R->>CB: Callback payload
            CB-->>C: Results
        end

    else No worker available
        B->>B: Add feature to pending_requests

        Note over B: Feature remains WAITING

        W->>R: Startup / heartbeat
        R->>B: Worker status
        B->>B: Register worker
        B->>R: Dispatch pending job
    end