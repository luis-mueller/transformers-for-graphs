from model.gd_transformer import GDTransformer
from model.edge_transformer import EdgeTransformer
from model.lwl_transformer import LWLTransformer
from model.transformer import FFN, ACTIVATION

MODEL_CLASS = {
    "GDT": GDTransformer,
    "ET": EdgeTransformer,
    "LWL": LWLTransformer,
}


MODEL_SIZE = {
    "10M": (10, 384, 768, 16),
    "12M": (12, 384, 384, 16),
    "24M": (12, 384, 768 * 2, 16),
    "16M": (16, 384, 384, 16),
    "50M": (12, 768, 768, 16),
    "90M": (24, 768, 768, 16),
    "160M": (24, 1024, 1024, 16),
}


def add_model_args(parser):
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASS.keys()))
    parser.add_argument("--model_size", type=str, choices=list(MODEL_SIZE.keys()))
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--ffn_dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--ffn", type=str, default="mlp", choices=list(FFN.keys()))
    parser.add_argument(
        "--activation", type=str, default="relu", choices=list(ACTIVATION.keys())
    )
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--use_edge_transform", action="store_true")


def load_model_size_from_args(args):
    return MODEL_SIZE[args.model_size]


def load_model_class_from_args(args):
    return MODEL_CLASS[args.model]


def load_model_from_args(args, modules, pe_encoder):
    num_layers, embed_dim, ffn_dim, num_heads = load_model_size_from_args(args)
    model_class = load_model_class_from_args(args)
    return model_class(
        modules,
        pe_encoder,
        num_layers,
        embed_dim,
        ffn_dim,
        num_heads,
        args.attention_dropout,
        args.ffn_dropout,
        args.bias,
        ffn=args.ffn,
        activation=args.activation,
        norm_first=args.norm_first,
        pooling=args.pooling,
    )
