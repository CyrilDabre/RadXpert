import torch
import numpy as np
from skimage import transform, io
from segment_anything import sam_model_registry
import torch.nn.functional as F

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "C:/Users/dabre/OneDrive/Documents/sem3/Capstone/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_medsam_model():
    model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=MedSAM_CKPT_PATH).to(device)
    model.eval()
    return model

@torch.no_grad()
def medsam_inference(model, image_tensor, bbox_coords):
    image_tensor = image_tensor.to(device)
    img_embed = model.image_encoder(image_tensor)

    box_torch = torch.tensor([bbox_coords], dtype=torch.float, device=img_embed.device).unsqueeze(1)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(MEDSAM_IMG_INPUT_SIZE, MEDSAM_IMG_INPUT_SIZE), mode="bilinear", align_corners=False)
    medsam_seg = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    return medsam_seg
