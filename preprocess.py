import argparse
from loguru import logger
from tasks import load_task, TASKS
from pe import load_transform_from_args, add_transform_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+", choices=TASKS.keys())
    parser.add_argument("--root", type=str, default=".")
    add_transform_args(parser)
    args = parser.parse_args()

    logger.info(
        f"Preprocessing data for tasks {args.tasks} with the following transforms: {args.transforms}"
    )

    pre_transform = load_transform_from_args(args)
    for task in args.tasks:
        load_task(
            task,
            args.root,
            pre_transform=pre_transform,
        )


if __name__ == "__main__":
    main()
