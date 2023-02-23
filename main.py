from Generator import *
from Discriminator import *
from Loss import *

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# 60000/128 = 469 steps in each epoch
# Each step will process 128 images except the last step, which will have a little less than 128 images
for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        # Discriminator
        disc_opt.zero_grad()

        cur_bs = len(real)  # Current batch size
        real = real.view(cur_bs, -1)  # 128 x 784
        real = real.to(device)

        disc_loss = calc_disc_loss(loss_func, gen, disc, cur_bs, real, z_dim)

        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Generator
        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(loss_func, gen, disc, cur_bs, z_dim)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        # Visualization and Stats
        mean_disc_loss += disc_loss.item() / info_step
        mean_gen_loss += gen_loss.item() / info_step

    fake_noise = gen_noise(cur_bs, z_dim)
    fake = gen(fake_noise)
    show(fake)
    show(real)
    print(f"Epoch: {epoch} / Gen loss: {mean_gen_loss} / disc_loss: {mean_disc_loss}")
    mean_gen_loss, mean_disc_loss = 0, 0
