import numpy as np

def gray_world_white_balance(bgr_img: np.ndarray, lambda_param: float = 0.2) -> np.ndarray:
    img = bgr_img.astype(np.float32) / 255.0
    mu_b, mu_g, mu_r = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    mu_gray = (mu_b + mu_g + mu_r) / 3.0
    target = 0.5 + lambda_param * mu_gray
    eps = 1e-8
    out = np.zeros_like(img)
    out[:,:,0] = np.clip(img[:,:,0] * (target / (mu_b + eps)), 0, 1)
    out[:,:,1] = np.clip(img[:,:,1] * (target / (mu_g + eps)), 0, 1)
    out[:,:,2] = np.clip(img[:,:,2] * (target / (mu_r + eps)), 0, 1)
    return (out * 255).astype(np.uint8)