import torch
from torch import nn
import torch.nn.functional as F
from xca import XCABlock


class ShapeTransformer(nn.Module):
    def __init__(self, token_size=64, disentangle_style=False, num_id_classes=None):
        super().__init__()

        self.encoder = ShapeTransformerEncoder(
            token_size=token_size,
        )
        self.decoder = ShapeTransformerDecoder(
            token_size=token_size,
            disentangle_style=disentangle_style
        )
        if disentangle_style:
            assert num_id_classes is not None
            self.id_classifier = nn.Linear(token_size//2, num_id_classes, bias=False)
        self.disentangle_style = disentangle_style
        self.token_size = token_size

    def forward(self, positions, offsets, decoder_positions=None):
        shape_code = self.encoder(positions, offsets)

        decoder_positions = positions if decoder_positions is None else decoder_positions
        return self.decoder(decoder_positions, shape_code)
        

class ShapeTransformerEncoder(nn.Module):
    """
    Comprises a sequence of 4 XCiT transformer blocks.
    """
    def __init__(self, token_size=64):
        super().__init__()

        self.pos_mlp = ResidualMLP(3, token_size//2, token_size//2, num_hidden=2)
        self.offset_mlp = ResidualMLP(3, token_size//2, token_size//2, num_hidden=2)
        self.shape_token = nn.Parameter(torch.randn(token_size),
                                        requires_grad=True)
        self.xca_blocks = nn.ModuleList([
            XCABlock(dim=token_size, lpi_kernel_size=1)
            for _ in range(4)
        ])

    def forward(self, positions, offsets):
        # Pass positions through position MLP
        latent_positions = self.pos_mlp(positions)  # B x N x 3 -> B x N x 32

        # Pass offsets through input offset MLP
        latent_offsets = self.offset_mlp(offsets)  # B x N x 3 -> B x N x 32

        # Concatenate corresponding latent positions and latent offsets
        # B x N x (32 + 32)
        tokens = torch.cat([latent_positions, latent_offsets], dim=-1)

        # Append shape_token to tokens
        # B x (N + 1) x 64
        tokens = torch.cat([
            tokens,
            self.shape_token.expand(len(tokens), 1, -1)
        ], dim=-2)

        # Pass tokens to transformer
        H = tokens.shape[-2]
        W = 1
        for xca_block in self.xca_blocks:
            tokens = xca_block(tokens, H, W)

        # Return output that corresponds to shape token
        return tokens[..., -1, :]


class ShapeTransformerDecoder(nn.Module):
    """
    Comprises a sequence of 4 XCiT transformer blocks with standard residual
    MLPs, but *with modulated input tokens*.
    """
    def __init__(self, token_size=64, disentangle_style=False):
        super().__init__()

        self.pos_mlp = ResidualMLP(3, token_size, token_size, num_hidden=2)
        self.offset_mlp = ResidualMLP(token_size, token_size, 3, num_hidden=2)

        self.style_mlp_shared = nn.Sequential(
            nn.Linear(token_size, token_size),
            nn.ReLU()
        )
        self.style_mlp_affine = nn.Linear(token_size,  token_size)

        self.disentangle_style = disentangle_style

        if self.disentangle_style:
            self.expression_mlp = ResidualMLP(token_size//2, token_size//2, token_size//2, num_hidden=1)

        self.xca_blocks = nn.ModuleList([
            XCABlock(dim=token_size, lpi_kernel_size=1)
            for _ in range(4)
        ])

    def forward(self, positions, shape_code):
        # Pass positions through position MLP
        latent_positions = self.pos_mlp(positions)

        if self.disentangle_style:
            #  Split shape code into expression and identity part
            # B x 64 -> (B x 32, B x 32)
            expr_code, id_code = torch.split(
                shape_code,
                shape_code.shape[-1] // 2,
                dim=-1
            )
            expr_code = self.expression_mlp(expr_code)
            shape_code = torch.cat([expr_code, id_code], dim=-1)

        # Pass shape code through 4 layers with shared weights
        for _ in range(4):
            shape_code = shape_code + self.style_mlp_shared(shape_code)

        style_code = self.style_mlp_affine(shape_code)

        # Pass tokens through transformer, each time modulating with style code
        tokens = latent_positions
        H = tokens.shape[-2]
        W = 1
        for xca_block in self.xca_blocks:
            tokens = tokens * style_code[..., None, :]
            tokens = xca_block(tokens, H, W)

        # Pass latent output offsets through OutputOffsetMLP
        offsets = self.offset_mlp(tokens)

        if self.disentangle_style:
            # Return identity code for supervision
            return offsets, id_code
        else:
            return offsets


class ResidualMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_hidden):
        super().__init__()
        self.in_layer = nn.Linear(in_features, hidden_features)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features)
            for _ in range(num_hidden)
        ])
        self.out_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.in_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            h = layer(x)
            x = F.relu(h + x)
        x = self.out_layer(x)
        return x