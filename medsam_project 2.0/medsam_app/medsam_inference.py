import torch
import numpy as np
from skimage import transform, io
from segment_anything import sam_model_registry
import torch.nn.functional as F

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "C:/Users/dabre/OneDrive/Documents/sem3/Capstone/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_medsam_model():
    model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=MedSAM_CKPT_PATH).to(device)
    model.eval()
    return model

@torch.no_grad()
def medsam_inference(medsam_model, image_path):
    img_np = io.imread(image_path)
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        img_embed = medsam_model.image_encoder(img_1024_tensor)

    box_1024 = np.array([[0, 0, 1024, 1024]])  # Full image box
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device).unsqueeze(1)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(img_3c.shape[0], img_3c.shape[1]), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg
