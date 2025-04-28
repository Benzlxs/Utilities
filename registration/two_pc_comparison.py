import numpy as np       # or import torch
rng = np.random.default_rng(seed=42)   # fixed seed → repeatable
import torch
import open3d as o3d
def umeyama(X, Y, with_scale=True):
    """
    X, Y: (N,3) numpy arrays of corresponding points
    Returns s, R, t such that Y ≈ s R X + t
    """
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_x, Y - mu_y
    Σ = (Yc.T @ Xc) / X.shape[0]       # covariance
    U, D, Vt = np.linalg.svd(Σ)
    R = U @ np.diag([1,1,np.sign(np.linalg.det(U) * np.linalg.det(Vt.T))]) @ Vt
    if with_scale:
        var_x = (Xc**2).sum() / X.shape[0]
        s = (D @ np.array([1,1,np.sign(np.linalg.det(U)*np.linalg.det(Vt.T))])) / var_x
    else:
        s = 1.0
    t = mu_y - s * R @ mu_x
    return s, R, t


pcd = o3d.io.read_point_cloud("/Users/li325/projects/public_dataset/active_percerption/1bf5f3d7a131657ca01094d9087d1cf485aee90d5d036a792815eaa7457c6495/mesh.ply")
tgt_pts = np.asarray(pcd.points)


pcd = o3d.io.read_point_cloud("/Users/li325/projects/Utilities/vis/dataset_preprocess/world_points_pcd.ply")
src_pts = np.asarray(pcd.points)


n_src, n_tgt = src_pts.shape[0], tgt_pts.shape[0]
if n_src > n_tgt: 
    keep = rng.choice(n_src, n_tgt, replace=False)
    src_pts = src_pts[keep]
else:
    keep = rng.choice(n_tgt, n_src, replace=False)
    tgt_pts = tgt_pts[keep]



s, R, t = umeyama(src_pts, tgt_pts, with_scale=True)
aligned_src = (s * (R @ src_pts.T)).T + t


def chamfer(p, q):
    diff_pq = (p[:, None, :] - q[None, :, :]).pow(2).sum(-1)
    diff_qp = diff_pq.t()                 # symmetric matrix
    return diff_pq.min(1)[0].mean() + diff_qp.min(1)[0].mean()

pcd1 = torch.as_tensor(aligned_src, dtype=torch.float32)
pcd2 = torch.as_tensor(tgt_pts,  dtype=torch.float32)
cd = chamfer(pcd1, pcd2)
print(cd)
