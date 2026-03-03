# config/default.yaml
input:
  format: "json"
  encoding: "utf-8"

processing:
  batch_size: 100
  parallel_workers: 4

output:
  format: "csv"
  include_metadata: true