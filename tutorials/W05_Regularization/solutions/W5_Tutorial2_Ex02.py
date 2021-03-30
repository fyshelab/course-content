def prune_l1_unstructured(model,prune_percent_weight,prune_percent_bias = 0):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percent_weight)
            prune.l1_unstructured(module, name='bias', amount=prune_percent_bias)

            print(
                "Sparsity in {}: {:.2f}%".format(name,
                    100. * float(torch.sum(module.weight == 0))
                    / float(module.weight.nelement())
                )
            )

##uncomment to run the test
test_model = Animal_Net()
prune_percent = 0.15
prune_l1_unstructured(test_model,0.15)
