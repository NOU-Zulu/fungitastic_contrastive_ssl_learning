import torch
import torch.nn.functional as F
import math

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    device = z1.device

    z = torch.cat([z1, z2], dim=0)     # 2N × D
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)  # 2N × 2N

    similarity /= temperature
    similarity = similarity.exp()  #for the exp in the loss function
    mask = torch.eye(2 * batch_size, device=device).bool()
    similarity = similarity.masked_fill(mask, 0)    #exclude self simmilarity


    L = torch.tensor(0.0, device=device)
    j = batch_size
    for i in range(batch_size):    #i,j  must iterate over all positive pairs
        L += -torch.log(
            similarity[i][j]  # loss of positive pair
            /torch.sum(similarity[i])) #sum of negatives plus the positive --> if negative is 0 the log loss goes to 1
        j+=1

    L*=2
    return L/(2*batch_size)

    '''
    mask = (torch.ones_like(similarity)
            - torch.diag_embed(torch.ones_like(similarity).diagonal(batch_size), batch_size)           #remove positives in upper triangle
            - torch.tril(torch.ones_like(similarity), 0)).bool() #remove upper triangle of tensor including main diag

    positives = torch.cat([
        torch.diag(similarity, batch_size),
        #torch.diag(similarity, -batch_size) duplicate so no value I think
    ], dim=0)

    negatives = similarity[mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)

    logits /= temperature

    return F.cross_entropy(logits, labels)


        img1A img2A img3A img4A img1B img2B img3B img4B
img1A     1     .     .     .     +     .     .     .
img2A     .     1     .     .     .     +     .     .
img3A     .     .     1     .     .     .     +     .
img4A     .     .     .     1     .     .     .     +
img1B     +     .     .     .     1     .     .     .
img2B     .     +     .     .     .     1     .     .
img3B     .     .     +     .     .     .     1     .
img4B     .     .     .     +     .     .     .     1


'''