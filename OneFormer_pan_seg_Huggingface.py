@torch.no_grad()
def visualize_oneformer_predictions_final(
    model, dataset, device="cuda",
    score_threshold=0.5, num_images=3
):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image
    import torch
    import os
    import random
    import torch.nn.functional as F

    # COLORS
    Wcircle_1 = np.array([255, 0, 0])      # red
    Wcircle_2  = np.array([255, 255, 0])  # Yellow
    Ycircle_1 = np.array([255, 0, 255])    # pink
    Ycircle_2 = np.array([0, 0, 255])    # blue
    Ycircle_Partial_1 = np.array([0, 255, 255]) # cyan
    Ycircle_Partial_2 = np.array([150, 75, 0]) # brown
    CAP = np.array([0, 100, 0])        # dark green
    Box = np.array([255, 200, 0])      # orange-yellow
    MARKER = np.array([128, 0, 255])    # purple-blue
    FLOOR = np.array([135, 206, 235])   # sky blue

    model.eval()
    model.to(device)

    plt.figure(figsize=(6 * num_images, 6))

    id2label = model.config.id2label

    for idx in range(num_images):
        random_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[random_idx]

        inputs = {
            k: v.unsqueeze(0).to(model.device) for k, v in sample.items() if k in ['pixel_values', 'pixel_mask', 'task_inputs']
        }

        # Load original image
        filename = dataset.files[random_idx]
        img_path = os.path.join(dataset.image_root, filename)
        orig = np.array(Image.open(img_path).convert("RGB"))
        H, W = orig.shape[:2]

        overlay = orig.copy()

        # -------------------------
        # MODEL
        # -------------------------
        outputs = model(**inputs)

        logits = outputs.class_queries_logits[0]
        masks_logits = outputs.masks_queries_logits[0]

        probs = logits.softmax(-1)[:, :-1]
        scores, labels = probs.max(-1)

        keep = scores > score_threshold

        scores = scores[keep]
        labels = labels[keep]
        masks_logits = masks_logits[keep]

        # -------------------------
        # RESIZE MASKS
        # -------------------------
        masks_logits = F.interpolate(
            masks_logits.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )[0]

        masks = masks_logits.sigmoid().cpu().numpy()

        alpha = 0.7

        # -------------------------
        # BUILD PANOPTIC MAP
        # -------------------------
        panoptic_map = np.zeros((H, W), dtype=np.int32)

        score_order = np.argsort(-scores.cpu().numpy())

        instance_id1 = 1
        instance_id2 = 1
        instance_id3 = 1
        things_instances = []

        for i in score_order:
            mask = masks[i] > 0.5
            cls = labels[i].item()

            if mask.sum() == 0:
                continue

            if cls == 0:  # Wcircle
                panoptic_map[mask] = 1000 + instance_id1
                things_instances.append((cls, instance_id1, mask, scores[i].item()))
                instance_id1 += 1

            elif cls == 1:  # Ycircle
                panoptic_map[mask] = 2000 + instance_id2
                things_instances.append((cls, instance_id2, mask, scores[i].item()))
                instance_id2 += 1

            elif cls == 2:  # Ycircle Partial
                panoptic_map[mask] = 3000 + instance_id3
                things_instances.append((cls, instance_id3, mask, scores[i].item()))
                instance_id3 += 1

            elif cls == 3:  # CAP
                panoptic_map[mask] = 4000

            elif cls == 4:  # Box
                panoptic_map[mask] = 5000

            elif cls == 5:  # Marker
                panoptic_map[mask] = 6000

        # Fill remaining as FLOOR
        panoptic_map[panoptic_map == 0] = 7000

        # -------------------------
        # COLOR OVERLAY (PANOPTIC)
        # -------------------------
        for val in np.unique(panoptic_map):
            mask = panoptic_map == val

            class_id = val // 1000
            inst_id = val % 1000

            if class_id == 1:  # Wcircle
                color = Wcircle_2 if inst_id % 2 == 0 else Wcircle_1

            elif class_id == 2:  # Ycircle
                color = Ycircle_2 if inst_id % 2 == 0 else Ycircle_1

            elif class_id == 3:  # Ycircle Partial
                color = Ycircle_Partial_2 if inst_id % 2 == 0 else Ycircle_Partial_1

            elif class_id == 4:  # CAP
                color = CAP

            elif class_id == 5:  # Box
                color = Box

            elif class_id == 6:  # Marker
                color = MARKER

            elif class_id == 7:  # Floor
                color = FLOOR

            else:
                continue

            overlay[mask] = (
                alpha * color + (1 - alpha) * overlay[mask]
            ).astype(np.uint8)

        # -------------------------
        # DRAW BOX + LABEL (ONLY Things)
        # -------------------------
        for cls, inst_id, mask, score in things_instances:
            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            if cls == 0:
                color = Wcircle_2 if inst_id % 2 == 0 else Wcircle_1

            elif cls == 1:
                color = Ycircle_2 if inst_id % 2 == 0 else Ycircle_1

            elif cls == 2:
                color = Ycircle_Partial_2 if inst_id % 2 == 0 else Ycircle_Partial_1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 3)

            label_name = id2label[cls]

            cv2.putText(
                overlay,
                f"{label_name} {score:.2f}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                4,
                cv2.LINE_AA
            )

        # -------------------------
        # SHOW
        # -------------------------
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(overlay)
        plt.axis("off")

    plt.tight_layout()
    plt.show()