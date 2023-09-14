import json

import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()
    model_path = os.path.join(root_path, "random_forest")

    # Serialize decision tree configuration
    model_config = os.path.abspath("random_forest_config.yml")
    model_config_rel_path_from_model_path = os.path.relpath(path=model_config,
                                                            start=model_path)

    with open(model_config, "w+") as fp:
        fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

    _ = mlflow.run(
        model_path,
        "main",
        parameters={
            "train_data": config["data"]["train_data"],
            "model_config": model_config_rel_path_from_model_path
        },
    )


if __name__ == "__main__":
    go()
