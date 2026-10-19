# -*- coding: utf-8 -*-
"""Day 5 作业 B：用 einsum 写注意力打分（P3 阶段的 VLA 预热）。

给定 Q:(B,L,D) 和 K:(B,S,D)，算出 scores:(B,L,S)，
再做缩放 + softmax，最后加一个因果掩码。
**全程零 for 循环。**

── 字母含义（把 ? 换成真实数字，把 ? 换成中文含义）────────────
    B = 2   一次处理几个样本（batch size）
    L = 4   query 序列长度：有 4 个位置要"提问"
    S = 6   key 序列长度：有 6 个位置可供"被查询"
    D = 8   每个 token 的特征维（head_dim / embedding 维）
    所以 scores 的形状是 (B,L,S) = (2,4,6)：2 个样本 × 4 个提问位置 × 6 个被查位置
────────────────────────────────────────────────────────────

用法（在**项目根目录**下）：

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/attention_scores.py
"""

from __future__ import annotations

import numpy as np

# ───────────────────────────────────────────────────────────────────
# 0. 超参数（固定值，方便对照输出）
# ───────────────────────────────────────────────────────────────────
B, L, S, D = 2, 4, 6, 8


# ═══════════════════════════════════════════════════════════════════
# 1. 打分：einsum
# ═══════════════════════════════════════════════════════════════════
def attention_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Q:(B,L,D), K:(B,S,D)  ->  scores:(B,L,S)

    要表达的是：对每个 b，把第 l 个 query 和第 s 个 key 的 D 维向量做点积。

    读法（一个字母一个字母读）：
        b  两边都有、输出也有  -> 不动它（batch 维原样带过去）
        d  两边都有、输出没有  -> 对齐相乘后求和消掉（这就是"点积"）
        l  只有 Q 有           -> 保留
        s  只有 K 有           -> 保留
    """

    return np.einsum("bld,bsd->bls", Q, K)

# ═══════════════════════════════════════════════════════════════════
# 2. softmax：沿最后一维归一化
# ═══════════════════════════════════════════════════════════════════
def softmax(x: np.ndarray) -> np.ndarray:
    """沿最后一维做 softmax。

    两个细节都不能省：
        ① 先减掉每行最大值：exp 遇到大数会溢出成 inf，inf/inf = nan。
           减最大值不改变结果（分子分母同乘一个常数），但把数值压回安全区。
        ② keepdims=True：保证分母形状是 (...,1) 而不是 (...)。
           否则 (B,L,S) 除以 (B,L) 会触发广播错误。
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════
# 3. 因果掩码：不许看未来
# ═══════════════════════════════════════════════════════════════════
def apply_causal_mask(x: np.ndarray) -> np.ndarray:
    """把上三角（未来位置）遮成 -inf，softmax 之后它们的权重就是 0。

    np.tril 取下三角（含对角线），结果是 (L,S) 的布尔矩阵：
        [[1 0 0 0 0 0]
         [1 1 0 0 0 0]
         [1 1 1 0 0 0]
         [1 1 1 1 0 0]]
    它只有 2 维，所以要加一个 batch 轴变成 (1,L,S)，让它能在 B 维上广播。
    """
    keep = np.tril(np.ones((x.shape[-2], x.shape[-1]), dtype=bool))[None, :, :]
    return np.where(keep, x, -np.inf)


# ═══════════════════════════════════════════════════════════════════
# 4. 验收：三个形状/数值检查
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((B, L, D))
    K = rng.standard_normal((B, S, D))

    scores = attention_scores(Q, K)
    print(f"[1] scores shape = {scores.shape}   期望 {(B, L, S)}")
    assert scores.shape == (B, L, S), "scores 形状不对，检查 TODO 1 的下标串"

    # 交叉验证：@ + transpose 应该得到同样的东西
    ref = Q @ K.transpose(0, 2, 1)
    print(f"[2] 与 Q @ K.transpose(0,2,1) 一致: {np.allclose(scores, ref)}")
    assert np.allclose(scores, ref), "数值不对，检查 TODO 1"

    # 缩放：除以 sqrt(D)
    scaled = scores / np.sqrt(D)

    # softmax：不加掩码
    w = softmax(scaled)
    print(f"[3] softmax 后每行和 = {w.sum(axis=-1)[0]}   期望全 1")
    assert np.allclose(w.sum(axis=-1), 1.0), "每行和不是 1，检查 TODO 2 的 axis/keepdims"

    # softmax：加因果掩码
    wm = softmax(apply_causal_mask(scaled))
    print(f"[4] 加掩码后第 0 行 = {np.round(wm[0, 0], 4)}")
    assert np.allclose(wm[0, 0, 1:], 0.0), "被遮掉的位置权重应该是 0，检查 TODO 3"
    assert np.allclose(wm.sum(axis=-1), 1.0), "加掩码后每行和仍应为 1"

    # 顺手验证：严格上三角（不含对角线）的权重必须全为 0
    # 注意 triu 默认含对角线，要写 k=1 才是"严格上三角"
    upper = np.triu(np.ones((L, S), dtype=bool), k=1)
    print(f"[5] 被遮掉的位置权重全为 0: {np.all(wm[:, upper] == 0)}")
    assert np.all(wm[:, upper] == 0), "上三角位置的权重应该全为 0"

    print("\n全部检查通过。再确认一遍：文件里没有任何 for 循环。")


if __name__ == "__main__":
    main()
