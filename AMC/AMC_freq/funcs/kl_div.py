import torch

def kl_tmp(mu_1, sigma_1, mu_2, sigma_2):
    # q: N(mu_1, sigma_1^2)
    # p: N(mu_2, sigma_2^2)
    assert len(mu_1) == len(sigma_1) == len(mu_2) == len(sigma_2)
    tmp_1 = torch.pow((mu_1 - mu_2),2)/(2*torch.pow(sigma_2,2))
    tmp_2_1 = torch.pow(sigma_1, 2)/torch.pow(sigma_2, 2)
    tmp_2_2 = torch.log(torch.pow(sigma_1, 2)/torch.pow(sigma_2, 2))
    return tmp_1 + 0.5*(tmp_2_1 -1 - tmp_2_2)


def kl(mu_vec, sigma_vec, sigma_prior):
    tmp_1 = torch.pow(mu_vec, 2)/(2*sigma_prior**2)
    tmp_2_1 = torch.pow(sigma_vec, 2)/(sigma_prior**2)
    tmp_2_2 = torch.log(torch.pow(sigma_vec, 2)/(sigma_prior**2))
    return torch.sum(tmp_1 + 0.5*(tmp_2_1 -1 - tmp_2_2))
