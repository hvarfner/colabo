import torch


def sample_offset(prior_location, bounds, offset_magnitude):
    dim = bounds.shape[1]
    offset_rnd = torch.normal(torch.zeros(dim), torch.ones(dim))
    offset_norm = offset_rnd / torch.linalg.norm(offset_rnd, 2)

    offset = (bounds[1, :] - bounds[0, :]) * offset_magnitude * torch.sqrt(torch.Tensor([len(prior_location)])) * offset_norm
    return torch.clamp(prior_location + offset, bounds[0, :], bounds[1, :])


def sample_random(bounds):
    dim = bounds.shape[1]
    prior_location = torch.clamp(
        (bounds[1, :] - bounds[0, :]) * torch.rand(dim) + bounds[0, :], bounds[0, :], bounds[1, :])
    return prior_location

# TODO add all the prior construction stuff in here, too
