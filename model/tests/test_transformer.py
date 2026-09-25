import pytest
import torch
import torch.nn as nn
from model.transformer import Transformer

TRANSFORMERENCODER_BWD_COMPAT_MAPPER = {
    "self_attn.in_proj_weight": "attention.c_attn.weight",
    "self_attn.in_proj_bias": "attention.c_attn.bias",
    "self_attn.out_proj": "attention.c_proj",
    "linear1": "ffn.w1",
    "linear2": "ffn.w2",
}


@pytest.mark.parametrize("NUM_LAYERS", [1, 2, 4, 6])
@pytest.mark.parametrize("NUM_HEADS", [1, 2, 4, 8])
@pytest.mark.parametrize("EMBED_DIM", [128, 192, 256])
@pytest.mark.parametrize("BIAS", [False, True])
def test_bwd_compat_TransformerEncoder(NUM_LAYERS, NUM_HEADS, EMBED_DIM, BIAS):
    ref_layer = nn.TransformerEncoderLayer(
        EMBED_DIM,
        NUM_HEADS,
        EMBED_DIM,
        0.0,
        batch_first=True,
        bias=BIAS,
    )
    ref_model = nn.TransformerEncoder(ref_layer, NUM_LAYERS)
    state_dict = ref_model.state_dict()

    model = Transformer(NUM_LAYERS, EMBED_DIM, EMBED_DIM, NUM_HEADS, 0.0, BIAS)
    model_state_dict = {}
    for k, v in state_dict.items():
        for m_key, m_rpl in TRANSFORMERENCODER_BWD_COMPAT_MAPPER.items():
            if m_key in k:
                k = k.replace(m_key, m_rpl)
                break
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict)

    x = torch.randn((2, 16, EMBED_DIM))
    attn_mask = torch.randn((2, NUM_HEADS, 16, 16))

    ref_x = ref_model(x, attn_mask.flatten(0, 1))
    x = model(x, attn_mask)
    torch.testing.assert_close(ref_x, x)
