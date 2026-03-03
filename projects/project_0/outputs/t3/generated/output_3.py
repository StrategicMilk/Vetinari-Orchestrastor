from project_name.pipeline import DataPipeline
from project_name.config import Config

# Load configuration
cfg = Config.from_yaml("config/production.yaml")

# Initialize pipeline
pipeline = DataPipeline(cfg)

# Run processing
results = pipeline.run(input_path="data/input.json")
print(f"Processed {len(results)} records.")