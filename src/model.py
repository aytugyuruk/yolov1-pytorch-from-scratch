import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# --- ALL THE CODE LOGIC BELOW IS TAKEN FROM THE OFFICIAL YOLOv1 PAPER ---
# -------------------------------------------------------------------------

# --- Convolutional block consisting of a convolutional layer, batch normalization, and LeakyReLU activation ---
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x
    

# --- Architecture configuration for the convolutional layers of YOLOv1 ---
# --- Each tuple represents (out_channels, kernel_size, stride, padding) for a convolutional layer
architecture_config = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

# --- YOLOv1 model definition ---
class YOLOv1(nn.Module):
    def __init__(self, in_channels,split_size, num_boxes, num_classes):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x)
        x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        return x

    # --- Method to create convolutional layers based on the architecture configuration ---
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        # --- You remember our config holds tuples, strings and lists ---
        # --- So we check for each of those types and create the appropriate layers ---
        for x in architecture:
            if type(x) == tuple:
                out_channels, kernel_size, stride, padding = x
                layers += [CNNBlock(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)]
                in_channels = out_channels

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1_out_channels, kernel_size1, stride1, padding1 = x[0]
                conv2_out_channels, kernel_size2, stride2, padding2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [CNNBlock(in_channels,conv1_out_channels,kernel_size=kernel_size1,stride=stride1,padding=padding1)]
                    layers += [CNNBlock(conv1_out_channels,conv2_out_channels,kernel_size=kernel_size2,stride=stride2,padding=padding2)]
                    in_channels = conv2_out_channels

        # --- Finally we return a sequential container of all the layers we created ---
        # --- The return looks like this: nn.Sequential(layer1, layer2, layer3,...) ---
        # --- Where each layer is either a CNNBlock or a MaxPool2d ---
        return nn.Sequential(*layers)
    
    # --- Fully connected layers for YOLOv1 ---
    # --- In the last linear layer >> (nn.Linear(4096, split_size * split_size * (num_boxes * 5 + num_classes) ---
    # --- We multiply by 5 Because each bounding box prediction consists of 5 values (x, y, w, h, confidence) ---
    def _create_fcs(self, split_size, num_boxes, num_classes):
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(4096, split_size * split_size * (num_boxes * 5 + num_classes))
        )
        return fc