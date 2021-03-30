def calculate_inference(N):

    total_time = 0.0
    model = Big_Animal_Net()
    model.eval()
    
    for i in range(N):
        X = torch.rand((1,3,32,32))
        with torch.no_grad():
            start_time = time.time()
            y = model(X)
            end_time = time.time()
            total_time += end_time - start_time

    print(f'Inference time of the above network is: {total_time/N}')

calculate_inference(1000)
