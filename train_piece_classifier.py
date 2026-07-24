import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = None


def get_chinese_font(size=22):
    global _FONT_CACHE
    if _FONT_CACHE is None:
        font_candidates = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for candidate in font_candidates:
            if os.path.exists(candidate):
                try:
                    _FONT_CACHE = ImageFont.truetype(candidate, size)
                    break
                except Exception:
                    pass
        if _FONT_CACHE is None:
            _FONT_CACHE = ImageFont.load_default()
    return _FONT_CACHE


def draw_chinese_text(img_bgr, text, center, ink_color_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(size=20)
    rgb = (int(ink_color_bgr[2]), int(ink_color_bgr[1]), int(ink_color_bgr[0]))
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = center[0] - w / 2.0 - bbox[0]
    y = center[1] - h / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 15 Classes: 0:"" (empty), 1-7: Red "K","A","B","N","R","C","P", 8-14: Black "k","a","b","n","r","c","p"
LABELS = ["", "K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}

CHINESE_CHARS = {
    "K": ["帥", "帅"],
    "A": ["仕", "士"],
    "B": ["相", "象"],
    "N": ["馬", "马"],
    "R": ["車", "车"],
    "C": ["炮", "砲"],
    "P": ["兵"],
    "k": ["將", "将"],
    "a": ["士", "仕"],
    "b": ["象", "相"],
    "n": ["馬", "马"],
    "r": ["車", "车"],
    "c": ["炮", "砲"],
    "p": ["卒"],
}


def extract_features(crop_bgr):
    """Extract a 464-dim robust feature vector from a 40x40 piece crop."""
    crop = cv2.resize(crop_bgr, (32, 32))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 1. Normalized grayscale vector (256-dim resized)
    small_gray = cv2.resize(gray, (16, 16)).flatten()
    small_gray -= small_gray.mean()
    norm = np.linalg.norm(small_gray)
    small_gray = small_gray / norm if norm else small_gray
    
    # 2. Red / Dark color ratio features
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_mask = (((hue < 10) | (hue > 170)) & (sat > 90) & (val > 70)).astype(np.float32)
    red_feat = cv2.resize(red_mask, (8, 8)).flatten()
    
    # 3. Horizontal and vertical stroke gradients (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_small = cv2.resize(mag, (12, 12)).flatten()
    mag_norm = np.linalg.norm(mag_small)
    mag_small = mag_small / mag_norm if mag_norm else mag_small
    
    # Concatenate features: 256 + 64 + 144 = 464 dims
    feat = np.concatenate([small_gray, red_feat, mag_small])
    return feat


def generate_synthetic_samples(num_per_class=300):
    X, y = [], []
    np.random.seed(42)

    for idx, label in enumerate(LABELS):
        for _ in range(num_per_class):
            # Create a 40x40 background with random wood/stone texture
            bg_color = np.random.randint(110, 190, size=3, dtype=np.uint8)
            img = np.full((40, 40, 3), bg_color, dtype=np.uint8)
            
            # Add random background noise/texture
            noise = np.random.randint(-15, 15, size=(40, 40, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Draw grid lines for empty or piece cells randomly
            if np.random.rand() > 0.4:
                cv2.line(img, (0, 20), (40, 20), (80, 80, 80), 1)
                cv2.line(img, (20, 0), (20, 40), (80, 80, 80), 1)

            if label != "":
                is_red = label.isupper()
                fill_color = (
                    (np.random.randint(180, 220), np.random.randint(200, 240), np.random.randint(220, 255))
                    if is_red else
                    (np.random.randint(190, 230), np.random.randint(190, 220), np.random.randint(170, 200))
                )
                ink_color = (
                    (np.random.randint(20, 50), np.random.randint(20, 50), np.random.randint(170, 230))
                    if is_red else
                    (np.random.randint(15, 45), np.random.randint(15, 45), np.random.randint(15, 45))
                )
                
                # Draw piece circle
                center = (20 + np.random.randint(-1, 2), 20 + np.random.randint(-1, 2))
                radius = np.random.randint(15, 18)
                cv2.circle(img, center, radius, fill_color, -1)
                cv2.circle(img, center, radius, ink_color, 2)

                # Add selection / move glow rings (simulating Tiantian Xiangqi / JJ Xiangqi highlights)
                if np.random.rand() > 0.4:
                    glow_color = (255, 255, 255) if np.random.rand() > 0.3 else (50, 255, 255)
                    cv2.circle(img, center, radius + 2, glow_color, np.random.randint(2, 5))

                # Draw character using PIL with real Chinese font
                chars = CHINESE_CHARS.get(label, [label.upper()])
                char_text = np.random.choice(chars)
                img = draw_chinese_text(img, char_text, center, ink_color)
            else:
                # Empty cell: randomly add route dots or selection circles
                if np.random.rand() > 0.7:
                    cv2.circle(img, (20, 20), np.random.randint(3, 8), (255, 255, 255), -1)

            feat = extract_features(img)
            X.append(feat)
            y.append(idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_neural_network():
    print("生成多皮肤/发光特效合成训练集...")
    X, y = generate_synthetic_samples(num_per_class=400)
    num_samples, dim = X.shape
    num_classes = len(LABELS)

    # Standardize input features (Zero-mean, unit variance)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    # Convert y to one-hot
    Y = np.zeros((num_samples, num_classes), dtype=np.float32)
    Y[np.arange(num_samples), y] = 1.0

    # 2-Layer Neural Network Architecture: Input(dim) -> Hidden(128) -> Output(15)
    hidden_dim = 128
    np.random.seed(42)
    W1 = np.random.randn(dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(num_classes, dtype=np.float32)

    # Adam Optimizer parameters
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.005
    epochs = 120
    batch_size = 64
    t = 0

    print(f"开始训练 2 层 Adam 微型神经网络 (特征维度: {dim}, 隐藏层: {hidden_dim}, 类别数: {num_classes})...")
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_shuffled, Y_shuffled = X_norm[indices], Y[indices]
        
        loss_sum = 0.0
        for start in range(0, num_samples, batch_size):
            t += 1
            end = start + batch_size
            xb, yb = X_shuffled[start:end], Y_shuffled[start:end]

            # Forward pass
            h = np.maximum(0, xb @ W1 + b1) # ReLU activation
            logits = h @ W2 + b2
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            loss = -np.mean(np.sum(yb * np.log(probs + 1e-7), axis=1))
            loss_sum += loss * len(xb)

            # Backward pass
            dlogits = (probs - yb) / len(xb)
            dW2 = h.T @ dlogits
            db2 = np.sum(dlogits, axis=0)

            dh = dlogits @ W2.T
            dh[h <= 0] = 0 # ReLU gradient
            dW1 = xb.T @ dh
            db1 = np.sum(dh, axis=0)

            # Adam update for W1, b1, W2, b2
            for param, dparam, m, v in [(W1, dW1, mW1, vW1), (b1, db1, mb1, vb1), (W2, dW2, mW2, vW2), (b2, db2, mb2, vb2)]:
                m[:] = beta1 * m + (1 - beta1) * dparam
                v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            h_all = np.maximum(0, X_norm @ W1 + b1)
            probs_all = h_all @ W2 + b2
            preds = np.argmax(probs_all, axis=1)
            acc = np.mean(preds == y) * 100.0
            print(f"Epoch {epoch+1:3d}/{epochs} - Loss: {loss_sum/num_samples:.4f} - Train Acc: {acc:.2f}%")

    model_path = Path(__file__).resolve().parent / "piece_classifier.npz"
    np.savez_compressed(
        model_path,
        W1=W1, b1=b1, W2=W2, b2=b2,
        mean=mean, std=std,
        labels=np.array(LABELS),
        dim=dim, hidden_dim=hidden_dim
    )
    print(f"神经网络模型权重已成功导出至: {model_path} (大小: {model_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    train_neural_network()
