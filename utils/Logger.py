import logging


class TrainingLogger(logging.Logger):
    def __init__(self, log_file="TrainLog.log"):
        super().__init__(__name__)
        self.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        self.total_ssim = 0
        self.ssim_list = []

        self.total_l1 = 0
        self.l1_list = []

        self.total_perceptual_loss = 0
        self.perceptual_loss_list = []

        self.total_recon_loss = 0
        self.recon_loss_list = []

        self.total_final_loss = 0
        self.final_loss_list = []

    def log_metrics(self, step, ssim, l1_loss, perceptual_loss, recon_loss, final_loss):
        self.total_ssim += ssim

        self.total_l1 += l1_loss

        self.total_perceptual_loss += perceptual_loss

        self.total_recon_loss += recon_loss

        self.total_final_loss += final_loss

        if step % 10 == 0:
            self.info(f"Step {step}: SSIM: {ssim:.4f}, L1 Loss: {l1_loss:.4f}, "
                      f"Perceptual Loss: {perceptual_loss:.4f}, Recon Loss: {recon_loss:.4f}, "
                      f"Final Loss: {final_loss:.4f}")

    def log_epoch(self, epoch, total):
        avg_ssim = self.total_ssim / total
        avg_l1 = self.total_l1 / total
        avg_perceptual_loss = self.total_perceptual_loss / total
        avg_recon_loss = self.total_recon_loss / total
        avg_final_loss = self.total_final_loss / total

        self.info(f"Epoch {epoch}: "
                  f"Average SSIM: {avg_ssim:.4f}, Average L1 Loss: {avg_l1:.4f}, "
                  f"Average Perceptual Loss: {avg_perceptual_loss:.4f}, "
                  f"Average Recon Loss: {avg_recon_loss:.4f}, "
                  f"Average Final Loss: {avg_final_loss:.4f}")

        self.ssim_list.append(avg_ssim)
        self.l1_list.append(avg_l1)
        self.perceptual_loss_list.append(avg_perceptual_loss)
        self.recon_loss_list.append(avg_recon_loss)
        self.final_loss_list.append(avg_final_loss)

        # Reset totals for next epoch
        self.total_ssim = 0
        self.total_l1 = 0
        self.total_perceptual_loss = 0
        self.total_recon_loss = 0
        self.total_final_loss = 0

    def save_loss_plot(self, filename="loss_plot.png"):
        import matplotlib.pyplot as plt

        plt.plot(self.ssim_list, label='SSIM')
        plt.plot(self.l1_list, label='L1 Loss')
        plt.plot(self.perceptual_loss_list, label='Perceptual Loss')
        plt.plot(self.recon_loss_list, label='Recon Loss')
        plt.plot(self.final_loss_list, label='Final Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()
