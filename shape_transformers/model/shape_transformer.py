import torch
from torch import nn

from .xca import XCABlock


class ShapeTransformer(nn.Module):
    def __init__(self, token_size=64, disentangle_style=False):
        super().__init__()

        self.encoder = ShapeTransformerEncoder(
            token_size=token_size,
        )
        self.decoder = ShapeTransformerDecoder(
            token_size=token_size,
            disentangle_style=disentangle_style
        )
        self.disentangle_style = disentangle_style
        self.token_size = token_size

    def forward(self, positions, offsets, decoder_positions=None):
        shape_code = self.encoder(positions, offsets)

        decoder_positions = (
            positions if decoder_positions is None
            else decoder_positions
        )
        return self.decoder(decoder_positions, shape_code)


class ShapeTransformerEncoder(nn.Module):
    """
    Comprises a sequence of 4 XCiT transformer blocks.
    """
    def __init__(self, token_size=64):
        super().__init__()

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, token_size//16),
            nn.ReLU(),
            nn.Linear(token_size//16, token_size//8),
            nn.ReLU(),
            nn.Linear(token_size//8, token_size//4),
            nn.ReLU(),
            nn.Linear(token_size//4, token_size//2),
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(3, token_size//16),
            nn.ReLU(),
            nn.Linear(token_size//16, token_size//8),
            nn.ReLU(),
            nn.Linear(token_size//8, token_size//4),
            nn.ReLU(),
            nn.Linear(token_size//4, token_size//2),
        )
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
    def __init__(
        self, token_size=64, disentangle_style=False,
        num_xca_blocks=4
    ):
        super().__init__()

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, token_size//8),
            nn.ReLU(),
            nn.Linear(token_size//8, token_size//4),
            nn.ReLU(),
            nn.Linear(token_size//4, token_size//2),
            nn.ReLU(),
            nn.Linear(token_size//2, token_size),
        )

        self.offset_mlp = nn.Sequential(
            nn.Linear(token_size, token_size//2),
            nn.ReLU(),
            nn.Linear(token_size//2, token_size//4),
            nn.ReLU(),
            nn.Linear(token_size//4, token_size//8),
            nn.ReLU(),
            nn.Linear(token_size//8, 3),
        )

        num_style_mlp_feats = (
            2 * token_size if disentangle_style
            else token_size
        )
        self.style_mlp = nn.Sequential(
            nn.Linear(num_style_mlp_feats, num_style_mlp_feats),
            nn.ReLU(),
            nn.Linear(num_style_mlp_feats, num_style_mlp_feats),
            nn.ReLU(),
            nn.Linear(num_style_mlp_feats, num_style_mlp_feats),
            nn.ReLU(),
            nn.Linear(num_style_mlp_feats, num_style_mlp_feats),
            nn.ReLU(),
        )
        self.style_mlp_heads = nn.ModuleList([
            nn.Linear(num_style_mlp_feats,  token_size)
            for i in range(num_xca_blocks)
        ])

        self.disentangle_style = disentangle_style

        if self.disentangle_style:
            self.expression_mlp = nn.Sequential(
                nn.Linear(token_size, token_size),
                nn.ReLU(),
                nn.Linear(token_size, token_size),
                nn.ReLU(),
                nn.Linear(token_size, token_size),
            )

        self.xca_blocks = nn.ModuleList([
            XCABlock(dim=token_size, lpi_kernel_size=1)
            for _ in range(num_xca_blocks)
        ])

    def forward(self, positions, shape_code, expr_code=None):
        # Pass positions through position MLP
        latent_positions = self.pos_mlp(positions)

        if self.disentangle_style:
            assert expr_code is not None
            expr_code = self.expression_mlp(expr_code)
            shape_code = torch.cat([expr_code, shape_code], dim=-1)

        # Pass shape code through 4 layer MLP
        style_code = self.style_mlp(shape_code)
        modulations = [
            style_head(style_code)
            for style_head in self.style_mlp_heads
        ]

        # Pass tokens through transformer, each time modulating with style code
        tokens = latent_positions
        H = tokens.shape[-2]
        W = 1
        for xca_block, modulation in zip(self.xca_blocks, modulations):
            tokens = modulation * style_code[..., None, :]
            tokens = xca_block(tokens, H, W)

        # Pass latent output offsets through OutputOffsetMLP
        offsets = self.offset_mlp(tokens)

        return offsets
