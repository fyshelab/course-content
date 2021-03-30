def grad_clipping(net, theta):  
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
    print(norm)

# Uncomment to test
sampleRNN2 = VanillaRNN(10, 50, 1000, 300).to(device)
x = torch.tensor([1, 2, 3]).view(1, -1).to(device)
output = sampleRNN2(x)
output.sum().backward()
grad_clipping(sampleRNN2, 10)
for p in sampleRNN2.parameters():
    assert torch.all(p.grad<=10)
print('Success!')