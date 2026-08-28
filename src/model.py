import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ComboLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() 
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.w_bce = weight_bce
        self.w_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (self.w_bce * bce_loss) + (self.w_dice * dice_loss)

def get_unet_model(device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_attention_type="scse"
    ).to(device)
    return model
