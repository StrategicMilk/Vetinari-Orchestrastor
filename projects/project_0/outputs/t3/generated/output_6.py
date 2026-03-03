graph LR
    A[Raw Input] --> B(Ingestor)
    B --> C(Validator)
    C --> D(Transformer)
    D --> E(Processor)
    E --> F(Emitter)
    F --> G[Output]