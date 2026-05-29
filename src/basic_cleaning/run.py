#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading input artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping price outliers outside [%s, %s]", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Drop rows outside proper NYC geolocation boundaries
    logger.info("Dropping rows outside of proper geolocation boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned dataframe
    df.to_csv("clean_sample.csv", index=False)

    # Log the cleaned dataset as an output artifact
    logger.info("Logging output artifact %s", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to consider",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price to consider",
        required=True,
    )

    args = parser.parse_args()

    go(args)
