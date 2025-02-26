from pydantic import BaseModel, PositiveInt


class DatasetConfig(BaseModel):
    num_samples: PositiveInt = 10000
    num_inputs: PositiveInt = 256
