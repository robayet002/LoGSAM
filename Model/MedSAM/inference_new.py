# -*- coding: utf-8 -*-
"""
Batch MedSAM inference:
- takes a folder of images
- takes a folder of JSON files (same name as image, *.json)
- for each image: read JSON, pick highest-score box, run MedSAM, save mask

Example:
python MedSAM_batch.py \
    --img_dir assets/test/images \
    --json_dir assets/test/bboxes \
    --out_dir assets/test/seg \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:0
python inference_new.py --img_dir /home/hpc/iwi5/iwi5357h/MedSAM/data/image --json_dir /home/hpc/iwi5/iwi5357h/MedSAM/data/label --out_dir /home/hpc/iwi5/iwi5357h/MedSAM/output --checkpoint work_dir/MedSAM/medsam_vit_b.pth --device cuda:0 --viz


"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import io, transform

from segment_anything import sam_model_registry  # same as your original code


# ------------------------
# visualization (unchanged)
# ------------------------
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def pick_best_box_from_json(json_path: Path):
    """
    Expect JSON like:
    {
      "labels": [...],
      "scores": [...],
      "bboxes": [[x1,y1,x2,y2], ...]
    }
    Return: np.array([[x1,y1,x2,y2]]) in image coordinates
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # minimal safety
    bboxes = data.get("bboxes", [])
    scores = data.get("scores", [])

    if not bboxes:
        raise ValueError(f"No bboxes in {json_path}")

    if not scores:
        # if no scores, just take the first bbox
        best_idx = 0
    else:
        # index of max score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

    best_box = np.array([bboxes[best_idx]], dtype=np.float32)  # shape (1, 4)
    return best_box


def load_and_preprocess_image(img_path: Path, device: str):
    img_np = io.imread(str(img_path))
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    return img_3c, img_1024_tensor, H, W


def main():
    parser = argparse.ArgumentParser(
        description="Batch MedSAM inference with image folder + JSON bboxes"
    )
    parser.add_argument("--img_dir", type=str, required=True, help="folder with images")
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="folder with json files (same basename as image)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="folder to save segmentation masks",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="path to MedSAM checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="save side-by-side visualization too (png)",
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # load model once
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    # list images
    exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]

    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    for img_path in img_paths:
        name = img_path.stem  # e.g. img_0001
        json_path = json_dir / f"{name}.json"

        if not json_path.exists():
            print(f"[WARN] JSON file missing for {img_path.name}, skipping...")
            continue

        print(f"Processing {img_path.name} ...")

        # 1. get best bbox from json
        box_np = pick_best_box_from_json(json_path)  # shape (1,4)

        # 2. load & preprocess image
        img_3c, img_1024_tensor, H, W = load_and_preprocess_image(img_path, device)

        # 3. forward image once
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        # 4. scale box to 1024
        # box: [x1, y1, x2, y2] in original image coords
        box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0

        # 5. run inference
        medsam_seg = medsam_inference(
            medsam_model, image_embedding, box_1024, H=H, W=W
        )

        # 6. save mask
        out_mask_path = out_dir / f"seg_{img_path.name}"
        #io.imsave(str(out_mask_path), medsam_seg, check_contrast=False)
        mask_to_save = (medsam_seg.astype(np.uint8)) * 255
        io.imsave(str(out_mask_path), mask_to_save, check_contrast=False)

        # 7. optional visualization
        if args.viz:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_3c)
            show_box(box_np[0], ax[0])
            ax[0].set_title("Input + Box")

            ax[1].imshow(img_3c)
            show_mask(medsam_seg, ax[1])
            show_box(box_np[0], ax[1])
            ax[1].set_title("MedSAM Segmentation")

            viz_path = out_dir / f"viz_{img_path.stem}.png"
            plt.tight_layout()
            plt.savefig(str(viz_path))
            plt.close(fig)

        print(f" -> saved {out_mask_path}")

    print("Done.")


if __name__ == "__main__":
    main()

