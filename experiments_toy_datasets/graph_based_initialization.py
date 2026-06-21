import numpy as np 
import torch 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# This file includes the graph-based initialization, i.e., Algorithm 1 in the thesis

def graph_init_curve(manifold, data_np, z0, z1, n_points,
                     k=15, N_edge=4, weight_floor=1e-8,
                     subsample=2000, seed=0, device='cpu'):
    
    if subsample is not None:
        # subsample data
        rng = np.random.default_rng(seed)
        choices = rng.choice(data_np.shape[0], subsample, replace=False)
        data_np = data_np[choices]

    z0_np = z0.detach().cpu().numpy().reshape(1, -1).astype(np.float32) # 1,D
    z1_np = z1.detach().cpu().numpy().reshape(1, -1).astype(np.float32)

    # construct vertex set V <-- data Union end/start points 
    V = np.concatenate([data_np.astype(np.float32), z0_np, z1_np], axis=0)
    N, D = V.shape
    qa, qb = N - 2, N - 1 # start and endpoint idx 

    # find nearest neighbors for each point 
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(V)
    # get k+1 nearest neighbors indices
    _, idx = nn.kneighbors(V) # N, k+1
    source_vertex_idx = np.repeat(np.arange(N), k) # shape N*k, 
    destination_vertex_idx = idx[:, 1:].reshape(-1) # drop first column, flatten row-wise, shape N*k, 

    source_vertex_coords  = torch.tensor(V[source_vertex_idx], dtype=torch.float32, device=device) # start point of each edge
    diff = torch.tensor(V[destination_vertex_idx] - V[source_vertex_idx], dtype=torch.float32, device=device) # difference vector 
    
    # get intermediate points
    s_u = torch.arange(1, N_edge, device=device, dtype=torch.float32) / (N_edge - 1)
    du = 1.0 / (N_edge - 1)

    with torch.no_grad():
        # compute all edge curves
        edge_curve = source_vertex_coords[:, None, :] + s_u[None, :, None] * diff[:, None, :] # num_edges, num_points_edge, D
        num_edges, num_points_edge, _ = edge_curve.shape
        # evaluate metric along each edge point for each edge 
        G = manifold.metric(edge_curve.reshape(-1, D)).reshape(num_edges, num_points_edge, D, D)
        df_b = diff[:, None, :].expand(num_edges, num_points_edge, D) # num_edges, num_points_edge, D
        # local metric length 
        seg_sqrt = torch.sqrt(torch.einsum('bti,btij,btj->bt', df_b, G, df_b).clamp(min=0)) # num_edges, num_points_edge
        # Riemannian length for each edge
        w = (seg_sqrt.sum(dim=1) * du).cpu().numpy()

    w = np.maximum(w, weight_floor) # clamp 

    # W[i, j]: riem. length of edge from vertex i to j 
    W = csr_matrix((w, (source_vertex_idx, destination_vertex_idx)), shape=(N, N))
    # run dijkstra algorithm 
    _, preds = shortest_path(W, directed=False, indices=qa, return_predecessors=True)

    # backtrack path 
    path = [qb]; current = qb
    while current != qa:
        current = int(preds[current])
        if current < 0:
            break
        path.append(current)
    path.reverse()
    # final path a to b 
    p_ab = V[path]

    # resample to uniform curve of length n_points 
    seg = np.linalg.norm(np.diff(p_ab, axis=0), axis=1) # euclid length of each segment 
    s = np.concatenate([[0.0], np.cumsum(seg)]) # 
    s /= max(s[-1], 1e-12) # scale to 0, 1
    s_new = np.linspace(0, 1, n_points)
    out = np.stack([np.interp(s_new, s, p_ab[:, d]) for d in range(D)], axis=-1)
    return torch.tensor(out, dtype=torch.float32) # return new initial curve and feed to discrete_geodesic 



