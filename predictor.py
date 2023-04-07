from torch import nn

class PREDICTOR(nn.Module):
    def __init__(self):
        super(PREDICTOR, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # output layer
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

PREDICTOR_FUNCTIONS={
    'PREDICTOR':PREDICTOR()
}


