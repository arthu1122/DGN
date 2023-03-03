def update_target_network_parameters(online_network, target_network, m):
    """
    linear update target_network params
    """
    for param_online, param_target in zip(online_network.parameters(), target_network.parameters()):
        param_target.data = param_target.data * m + param_online.data * (1. - m)


def initializes_target_network(online_network, target_network):
    """
    copy params form online_network to target_network and set target_network ``requires_grad=False``
    """
    for param_online, param_target in zip(online_network.parameters(), target_network.parameters()):
        param_target.data.copy_(param_online.data)
        param_target.requires_grad = False
