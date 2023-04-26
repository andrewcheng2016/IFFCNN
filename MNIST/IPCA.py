import numpy as np
np.set_printoptions(suppress=True)
import torch



def IPCA(Q_old=None, n=None, x_new=None, mean_old=None, sqs_old=None, use_gpu=True):
    """
    IncrementalPCA

    :param Q_old: the covariance matrix at the previous step (set it to zero at the first step)
    :param n: the current step counter
    :param x_new: the new sample
    :param mean_old: the mean at the previous step
    :param sqs_old: the sum of squares at the previous step
    :param use_gpu: use gpu or not
    :return: returns the coefficients coeff and the vector of PC's latent
    Q_new, mean_new, sqs_new are used as inputs for the next steps respectively for Q_old, mean_old, sqs_old

    References
    ----------
    V. Lippi and G. Ceccarelli, “Incremental principal component analysis exact implementation and continuity corrections,” in ICINCO, 2019.
    ----------
    """

    # Place the array on GPU if use_gpu is True
    device = 'cuda' if(use_gpu == True) else 'cpu'

    Q_old = torch.from_numpy(Q_old).to(device)
    n = torch.tensor(n).to(device)
    mean_old = torch.from_numpy(mean_old).to(device)
    sqs_old = torch.from_numpy(sqs_old).to(device)
    x_new = torch.from_numpy(x_new).to(device)

    Q_new, mean_new, sqs_new, sigma_new, z_new, n, gamma = UpdateQ(Q_old, x_new, n, mean_old, sqs_old)
    covariance_matrix = Q_new

    u, s, v = torch.svd(covariance_matrix)
    coeff = v
    latent = s

    return coeff, latent, covariance_matrix, mean_new, sqs_new, sigma_new, z_new, n, gamma




def UpdateQ(Q_old, x_new, n, mean_old, sqs_old):
    """
    UPDATEQ updates the non-normalized covariance matrix for the dataset X, with a new sample XNEW.
    The non normalized covariance matrix is the covariance matrix multiplied by N-1, where N is the number of samples.

    :param Q_old: the covariance matrix at the previous step (set it to zero at the first step)
    :param x_new: the new sample
    :param n: current step counter
    :param mean_old: the mean at the previous step
    :param sqs_old: the sum of squares at the previous step
    :return: Q_new: the new non-normalized covariance matrix and MNEW the row vector of data series means. Both values are used for incremental update.

    References
    ----------
    V. Lippi and G. Ceccarelli, “Incremental principal component analysis exact implementation and continuity corrections.” in ICINCO, 2019.
    ----------

    """


    number_of_x_new = x_new.shape[0]
    mean_new = ((mean_old*n)+(torch.mean(x_new,axis=0)*(x_new.shape[0])))/(n+x_new.shape[0])
    sigma_old = torch.sqrt(((sqs_old - (n) * (torch.square(mean_old))) / (n - 1)))
    sqs_new = sqs_old + torch.sum(torch.square(x_new),axis=0)
    sigma_new = torch.sqrt(((sqs_new-(n+number_of_x_new)*(torch.square(mean_new)))/(n+number_of_x_new-1)))
    gamma = torch.diag(1 / sigma_new)
    xi = torch.diag(sigma_old)
    R = gamma * xi

    if(x_new.shape[0] <= 1):
        x = torch.matmul(torch.subtract(x_new,mean_new).flatten(), gamma)
    else:
        x = torch.matmul(torch.subtract(x_new,mean_new), gamma)

    d = mean_old - mean_new
    d_conjugate_transpose = torch.unsqueeze(d,0).T
    x_conjugate_transpose = x.T

    if(x_new.shape[0] == 1):
        Q_new = torch.matmul(torch.matmul(R, Q_old), R) + torch.matmul(gamma, torch.matmul(
            (n * torch.matmul(d_conjugate_transpose, torch.unsqueeze(d,0))), gamma)) + torch.matmul(
            x_conjugate_transpose, torch.unsqueeze(d,0))


    else:
        Q_new = torch.matmul(torch.matmul(R, Q_old), R) \
                + torch.matmul(gamma, torch.matmul((n * torch.matmul(d_conjugate_transpose, torch.unsqueeze(d,0))), gamma))\
                + torch.matmul(x_conjugate_transpose,x)



    return Q_new, mean_new, sqs_new, sigma_new, x, n+x_new.shape[0], gamma
