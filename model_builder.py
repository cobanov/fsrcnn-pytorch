import torch
import torch.nn as nn

"""
Overall structure: We can connect the above five parts to form a complete FSRCNN network as

# d: 56, s: 12
First Part:
    - Conv(5, d, 1) -> PReLU

Mid Part:
    - Conv(1, s, d)-> PReLU
    - m * Conv(3, s, s)-> PReLU
    - Conv(1, d, s) -> PReLU

Last Part:
    - DeConv(9, 1, d)
"""


class FSRCNN(nn.Module):
    def __init__(self, scale, in_channels=1, d=56, s=12, m=4) -> None:
        super().__init__()

        # First Part
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=5),
            nn.PReLU(),
        )

        ## Mid Part
        mid_layers = [
            nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1),
            nn.PReLU(s),
        ]
        for _ in range(m):
            mid_layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=s))
            mid_layers.append(nn.PReLU(s))

        self.mid_part = nn.Sequential(*mid_layers)

        # Last Part
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=9)

    def forward(self, x: torch.Tensor):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return None


print(FSRCNN(4))
