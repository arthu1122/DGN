def update_target_network_parameters(online_network, target_network, m):
    for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


def initializes_target_network(online_network, target_network):
    for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
