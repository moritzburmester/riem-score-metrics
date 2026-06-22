import torch
import numpy 
from rae import RAE2
from dataset import get_dataloader
from dataloader import AlphaNumV2

def run_training_loop(dataset, model, batch_size=128, lr=1e-4, alpha=1, lam=0.1, num_epochs=50, num_iters=500):

    device = next(model.parameters()).device         

    data = dataset.image  # (7, 180, 1, 32, 32): 7 letters, 180 angles (2 deg), 1 ch, 32x32
    data = data.to(device)                             
    n_letter = data.shape[0]
    n_rotation = data.shape[1]
    step = 360.0 / n_rotation                         
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   

    history = []
    for epoch in range(num_epochs):
        for it in range(num_iters):                    
            optimizer.zero_grad()

            L = torch.randint(high=n_letter,   size=(batch_size,), device=device)
            a = torch.randint(high=n_rotation, size=(batch_size,), device=device)

            pos_mag  = torch.randint(low=1, high=6,  size=(batch_size,), device=device)  
            pos_sign = torch.randint(low=0, high=2,  size=(batch_size,), device=device) * 2 - 1
            pos_off  = pos_mag * pos_sign

            neg_mag  = torch.randint(low=15, high=90, size=(batch_size,), device=device)  
            neg_sign = torch.randint(low=0, high=2,   size=(batch_size,), device=device) * 2 - 1
            neg_off  = neg_mag * neg_sign

            p = (a + pos_off) % n_rotation
            n = (a + neg_off) % n_rotation

            xa = data[L, a] * 2 - 1
            xp = data[L, p] * 2 - 1
            xn = data[L, n] * 2 - 1

            za = model.encode(xa)
            zp = model.encode(xp)
            zn = model.encode(xn)

            L_rec = ((model.decode(za) - xa) ** 2).mean()    
           
            didx_p = torch.min(torch.abs(a - p), n_rotation - torch.abs(a - p))
            didx_n = torch.min(torch.abs(a - n), n_rotation - torch.abs(a - n))
            delta_thetap = didx_p.float() * step
            delta_thetan = didx_n.float() * step

            dist_p = torch.norm(za - zp, dim=1)          
            dist_n = torch.norm(za - zn, dim=1)
            L_triplet = ((dist_p - alpha * delta_thetap) ** 2
                       + (dist_n - alpha * delta_thetan) ** 2).mean()

            # add with reg term lam
            L_total = L_rec + lam * L_triplet

            L_total.backward()
            optimizer.step()

            history.append({'epoch': epoch, 'it': it,
                        'rec': L_rec.item(),
                        'triplet': L_triplet.item(),
                        'total': L_total.item()})
            if it % 100 == 0:                            
                print(f'Epoch: {epoch}, iter: {it}, '
                      f'loss_rec: {L_rec.item():.4f}, '
                      f'loss_triplet: {L_triplet.item():.4f}, '
                      f'total: {L_total.item():.4f}')
    
    model.eval()
    return model, history

if __name__ == '__main__':
    # hyperparmeters (from Betune et al., https://arxiv.org/pdf/2505.18230, E.2)
    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = '/home/moritz.burmester/riemannian-score-metrics/experiments_rotatedcharacters/alphanum'
    lr = 1e-4 
    B = 128
    alpha = 1
    lam = 0.1
    num_epochs = 7
    num_iters = 500

    ds_alphanum = AlphaNumV2(data_root = data_root, sequential=False, rot_dist='uniform')
    model = RAE2(in_ch=1, nb_feature=128, z_dim=64).to(device)

    model, history = run_training_loop(dataset=ds_alphanum, model=model, batch_size=B, lr=lr,
     alpha=alpha, lam=lam, num_epochs=num_epochs, num_iters=num_iters)

    print("Finished training, saving model at ./rae_triplet.pt...")
    torch.save(model.state_dict(), 'rae_triplet.pt')

