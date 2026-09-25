from evaluation.supervised import evaluate_supervised
from evaluation.transfer import evaluate_transfer


EVALUATIONS = {
    "supervised": evaluate_supervised,
    "transfer": evaluate_transfer,
}


def add_evaluation_args(parser):
    parser.add_argument(
        "--evaluation", type=str, default="supervised", choices=EVALUATIONS
    )


def load_evaluation(evaluation):
    return EVALUATIONS[evaluation]


def load_evaluation_from_args(args):
    return load_evaluation(args.evaluation)
