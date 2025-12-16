"""TODO"""

import time
import pickle
import os

import nrrd
import numpy as np
import torch
from matplotlib import pyplot as plt

from micronet import (
    load_img,
    load_annotation,
    MicroRNet,
)
from perturbimage import make_size_map, perturb_image


class TrainingVolume:
    def __init__(self, name, y_crop, device="cpu", error_bias=1.):
        self.name = name
        self.device = device
        self.error_bias = error_bias
        print(f"Volume {name}: Loading images.")
        self.img, self.header = load_img("E5D_" + name, device=device)
        seg = load_annotation("annotation_" + name + "_iso_full")
        self.img = self.img[:, :y_crop, :]
        seg = seg[:, :y_crop, :]
        os.makedirs("size_cache_map", exist_ok=True)
        smap_path = "size_map_cache/" + name + ".nrrd"
        if os.path.exists(smap_path):
            self.smap, _ = nrrd.read(smap_path, index_order="C")
        else:
            print("Making size map...")
            self.smap = make_size_map(seg)
            nrrd.write(smap_path, self.smap, index_order="C")
        self.seg = seg.astype(np.float32)
        self.mask_t = torch.tensor(seg[None, None, :, :, :] == 1,
                               dtype=torch.bool, device=device)
        self.mask_f = ~self.mask_t
        self.nt = torch.sum(self.mask_t)

        # Logs
        self.steps = []
        self.log_f = []
        self.log_cost = []
        self.log_precision = []
        self.log_recall = []

    def prep_batch(self, bs):
        batch = self.img.repeat(bs, 2, 1, 1, 1)
        size_ratios = np.random.uniform(1.0, 1.4, (bs,))
        nqs = np.random.uniform(0.02, 0.04, (bs,))
        for i in range(bs):
            p_seg = perturb_image(self.seg, self.smap, size_ratios[i], nqs[i])
            batch[i, 1, :, :, :] = torch.tensor(p_seg, dtype=torch.float32,
                                                device=self.device)

        # Need to normalize the distribution.
        # The annotation is a Bernoulli distribution with p ~ 0.05.
        # Mean is p. Variance is p*(1-p).
        batch[:, 1, :, :, :] *= 1/np.sqrt(0.05 * (1 - 0.05))
        batch[:, 1, :, :, :] -= 0.05
        return batch

    def cost(self, yp):
        global rpbal  # Where rpbal is defined as beta^2.
        
        tp = torch.sum(yp * self.mask_t)
        fp = torch.sum(yp * self.mask_f)
        fn = self.nt - tp
        if rpbal == 1.:
            fse = 1 - 2*tp / (2*tp + fn + fp)
        else:
            stp = (1 + rpbal)*tp
            fse = 1 - stp / (stp + rpbal*fn + fp)
        ypb = yp > 0.5
        tpb = torch.sum(ypb * self.mask_t)
        precision = tpb / torch.sum(ypb)
        recall = tpb / (self.nt * yp.shape[0])
        return fse, precision, recall

    def log_step(self, step: int, cost, precision, recall):
        f_score = 2 / (1/pre + 1/rec)
        self.steps.append(step)
        self.log_f.append(f_score.item())
        self.log_cost.append(cost.item())
        self.log_precision.append(precision.item())
        self.log_recall.append(recall.item())

    def logs_dict(self):
        return {
            "steps": self.steps,
            "cost": self.log_cost,
            "precision": self.log_precision,
            "recall": self.log_recall,
            "f_score": self.log_f
        }


def log_everything():
    training_logs = {
        "time": log_time,
        "learning_rate": log_lr,
        "validation_f_score": log_val,
        "training_volumes": {tv.name: tv.logs_dict() for tv in training_volumes}
    }
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/train_{model.name}_{train_ver}.pkl", "wb") as f:
        pickle.dump(training_logs, f)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Training on", device)
RAND_SEED = 43283
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED + 34921)
#torch.autograd.set_detect_anomaly(True)

train_ver = "diceb0.5"

# Adjusts how much precision is favored.
rpbal: float = 1.

# Learning rate
LR_MIN = 1e-4
LR_MAX = 2e-3
LR_DECAY = 1/500  # Rate of learning decay per step.
LR_PERIOD = 400  # Number of steps per learning cycle.
lr_override = None

validate_every = 50  # Iterations
finished = False  # Flag to exit the training process.
n_batches = 1  # How many training runs to do.
batch_iters = 3  # How many gradient descent iterations to make over each batch.
step = 0  # Total number of gradient descent iterations completed.
# Logging stuff:
secs_cumulative = 0
log_time = []
log_lr = []
log_val = []


def main():
    chkpt_dir = "checkpoints"
    os.makedirs(chkpt_dir, exist_ok=True)

    model = MicroRNet().to(device)
    print("Model:", model.name)
    training_volumes = np.array([
        TrainingVolume("082616_00_T46", 160, device),
        TrainingVolume("082616_01_T18", 160, device),
        TrainingVolume("082616_02_T45", 160, device),
        TrainingVolume("082616_04_T43", 150, device),
        TrainingVolume("082616_05_T16", 150, device),
        TrainingVolume("082616_06_T41", 150, device),
        TrainingVolume("082616_07_T15", 150, device),
        TrainingVolume("082616_08_T39", 160, device),
        TrainingVolume("082616_09_T14", 170, device),
        TrainingVolume("082616_10_T38", 170, device),
        TrainingVolume("082616_11_T14", 175, device),
        TrainingVolume("082616_12_T39", 190, device),
        TrainingVolume("082616_13_T14", 190, device),
    ], object)

    print("Preparing training set batches...")
    val = TrainingVolume("082616_03_T17", 170, device)
    val_batch = val.prep_batch(3)
    #test = TrainingVolume("082616_07_T40", 150, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MIN)

    while True:
        while True:
            r = input("> ")
            try:
                a, b = r.split()
                n_batches = int(a)
                batch_iters = int(b)
                print(f"Looping {n_batches} times, {batch_iters} steps each.")
                break
            except:
                pass
            if r.startswith("load "):
                ver = r.split()[1]
                torch.save(model.state_dict(),
                    f"{chkpt_dir}/{model.name}_{ver}.pt")
            elif r.startswith("load "):
                ver = r.split()[1]
                try:
                    model.load_state_dict(torch.load(
                        f"{chkpt_dir}/{model.name}_{ver}.pt"))
                except:
                    print(f"Couldn't load from {ver}")
            elif r == "log":
                log_everything()
            elif r.startswith("lr "):
                try:
                    lr_override = float(r.split()[1])
                except:
                    print("Couldn't parse the learning rate. :/")
            elif r.startswith("rp "):
                try:
                    rpbal = float(r.split()[1])
                except:
                    print("Couldn't parse the precision ratio/recall. :/")
            elif r == "finish":
                log_everything()
                finished = True
                break
            else:
                print("I didn't catch that.")

        if finished:
            break

        for ib in range(n_batches):
            last_time = time.time()
            print(f"Preparing batch {ib}/{n_batches} of size {len(training_volumes)}:")
            batch_train = [v.prep_batch(1) for v in training_volumes]
            gen_time = time.time() - last_time
            print(f"Generation took {gen_time:.1f} sec.")
            for ie in range(batch_iters):
                batch_order = np.arange(len(training_volumes))
                np.random.shuffle(batch_order)
                for iv in batch_order:
                    tv = training_volumes[iv]
                    yp = model(batch_train[iv])
                    J, pre, rec = tv.cost(yp)

                    if lr_override:
                        lr = lr_override
                    else:
                        lr = (LR_MIN + LR_MAX * (1 - abs(np.cos(step * 2*np.pi/LR_PERIOD)))
                              / np.sqrt(1 + step*LR_DECAY))
                    optimizer.param_groups[0]["lr"] = lr
                    
                    print(f"{step}, {ie}/{batch_iters}, lr={lr:.1e}"
                          f", set: {tv.name}"
                          f", pre|rec: {pre.item():.3f}|{rec.item():.3f}"
                          f", loss: {J.item():.4f}")

                    if step % validate_every == 0:
                        print("Validating... ", end="")
                        _, v_pre, v_rec = val.cost(model(val_batch))
                        fv = 2/(1/v_pre + 1/v_rec)
                        log_val.append(fv)
                        print(f"{fv:.3f}")

                    optimizer.zero_grad()
                    J.backward()
                    optimizer.step()
                    step += 1
                    
                    next_time = time.time()
                    delta_time = next_time - last_time
                    last_time = next_time
                    secs_cumulative += delta_time
                    
                    torch.save(model.state_dict(),
                               f"{chkpt_dir}/{model.name}_{train_ver}_{step}.model")
                    log_time.append(secs_cumulative)
                    log_lr.append(lr)
                    tv.log_step(step, cost=J, precision=pre, recall=rec)

        log_everything()

        fig, axes = plt.subplots(1, 2)
        legend = []
        for tv in training_volumes:
            if tv.steps:
                legend.append(tv.name)
                axes[0].plot(tv.steps, [-np.log10(1 - f) for f in tv.log_f], ".-")
                axes[1].plot(tv.steps, tv.log_cost, ".-")
        axes[0].plot(np.arange(len(log_val)) * validate_every,
                     [-np.log10(1 - f) for f in log_val], "k.-")
        legend.append("Cross validation")
        ##axes[0].set_ylim([0, 1])
        axes[0].set_title("-log10(1 - F_Score)")
        axes[1].legend(legend)
        axes[1].set_title("Cost")
        for a in axes:
            a.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
