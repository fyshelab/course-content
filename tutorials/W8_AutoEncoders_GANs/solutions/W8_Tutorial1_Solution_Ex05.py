class ConvVAE(nn.Module):
    def __init__(self, K, num_filters=32, filter_size=5):
        super(ConvVAE, self).__init__()
        
        # With padding=0, the number of pixels cut off from each image dimension
        # is filter_size // 2. Double it to get the amount of pixels lost in
        # width and height per Conv2D layer, or added back in per 
        # ConvTranspose2D layer.
        filter_reduction = 2 * (filter_size // 2)

        # After passing input through two Conv2d layers, the shape will be
        # 'shape_after_conv'. This is also the shape that will go into the first
        # deconvolution layer in the decoder
        self.shape_after_conv = (num_filters,
                                 my_dataset_size[1]-2*filter_reduction,
                                 my_dataset_size[2]-2*filter_reduction)
        flat_size_after_conv = self.shape_after_conv[0] \
            * self.shape_after_conv[1] \
            * self.shape_after_conv[2]

        # Define the recognition model (encoder or q) part
        self.q_bias = BiasLayer(my_dataset_size)      
        self.q_conv_1 = nn.Conv2d(my_dataset_size[0], num_filters, 5)
        self.q_conv_2 = nn.Conv2d(num_filters, num_filters, 5)
        self.q_flatten = nn.Flatten()
        self.q_fc_phi = nn.Linear(flat_size_after_conv, K+1)

        # Define the generative model (decoder or p) part
        self.p_fc_upsample = nn.Linear(K, flat_size_after_conv)
        self.p_unflatten = nn.Unflatten(-1, self.shape_after_conv)
        self.p_deconv_1 = nn.ConvTranspose2d(num_filters, num_filters, 5)
        self.p_deconv_2 = nn.ConvTranspose2d(num_filters, my_dataset_size[0], 5)
        self.p_bias = BiasLayer(my_dataset_size)

        # Define a special extra parameter to learn scalar sig_x for all pixels
        self.log_sig_x = nn.Parameter(torch.zeros(()))
    
    def infer(self, x):
        """Map (batch of) x to (batch of) phi which can then be passed to
        rsample to get z
        """
        s = self.q_bias(x)
        s = F.relu(self.q_conv_1(s))
        s = F.relu(self.q_conv_2(s))
        flat_s = s.view(s.size()[0], -1)
        phi = self.q_fc_phi(flat_s)
        return phi

    def generate(self, zs):
        """Map [b,n,k] sized samples of z to [b,n,p] sized images
        """
        # Note that for the purposes of passing through the generator, we need
        # to reshape zs to be size [b*n,k]
        b, n, k = zs.size()
        s = zs.view(b*n, -1)
        s = F.relu(self.p_fc_upsample(s)).view((b*n,) + self.shape_after_conv)
        s = F.relu(self.p_deconv_1(s))
        s = self.p_deconv_2(s)
        s = self.p_bias(s)
        mu_xs = s.view(b, n, -1)
        return mu_xs
    
    def forward(self, x):
        # VAE.forward() is not used for training, but we'll treat it like a
        # classic autoencoder by taking a single sample of z ~ q
        phi = self.infer(x)
        zs = rsample(phi, 1)
        return self.generate(zs).view(x.size())

    def elbo(self, x, n=1):
        """Run input end to end through the VAE and compute the ELBO using n
        samples of z
        """
        phi = self.infer(x)
        zs = rsample(phi, n)
        mu_xs = self.generate(zs)
        return log_p_x(x, mu_xs, self.log_sig_x.exp()) - kl_q_p(zs, phi)

def train_vae(vae, dataset, epochs=10, n_samples=16):
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=1e-6)
    elbo_vals = []
    vae.to(DEVICE)
    vae.train()
    loader = DataLoader(dataset, batch_size=100, shuffle=True, pin_memory=True)
    for epoch in trange(epochs, desc='Epochs'):
        for im, _ in tqdm(loader, total=len(dataset)//100, desc='Batches', leave=False):
            im = im.to(DEVICE)
            opt.zero_grad()
            loss = -vae.elbo(im)
            loss.backward()
            opt.step()

            elbo_vals.append(-loss.item())
    vae.to('cpu')
    vae.eval()
    return elbo_vals

vae = ConvVAE(K=K)
elbo_vals = train_vae(vae, my_dataset, n_samples=10)

print(f'Learned sigma_x is {torch.exp(vae.log_sig_x)}')

plt.figure()
plt.plot(elbo_vals)
plt.xlabel('Batch #')
plt.ylabel('ELBO')
plt.show()
