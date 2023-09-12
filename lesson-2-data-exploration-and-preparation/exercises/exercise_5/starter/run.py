#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with wandb.init(project="exercise_5", job_type="process_data") as run:
        ## YOUR CODE HERE
        # Fetch artifact
        logger.info(f'Fetch artifact {args.input_artifact}')
        artifact = run.use_artifact(args.input_artifact)
        df = pd.read_parquet(artifact.file())
        # Preprocess data
        logger.info('Do preprocessing')
        df = (df
            # drop duplicates
            .drop_duplicates()
            .reset_index(drop=True)
            # fill nans
            .fillna(dict(title='',
                        song_name=''))
            # Create text_feaure
            .assign(text_feature=lambda df: df.title + ' ' + df.song_name))
        # Save clean data
        df.to_csv(args.artifact_name)
        # Upload artifact
        logger.info(f'Upload artifact {args.artifact_name}')
        artifact = wandb.Artifact(name=args.artifact_name,
                                description=args.artifact_description,
                                type=args.artifact_type)
        artifact.add_file(args.artifact_name)
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
