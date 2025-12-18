"""Interactive tool for training basic 3D Micro Networks."""

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
    MicroNet,
)


class TrainingVolume:
    def __init__(self, name, y_crop, device="cpu", error_bias=1.):
        self.name = name
        self.DEVICE = DEVICE
        self.error_bias = error_bias
        print(f"Volume {name}: Loading images.")
        self.img, self.header = load_img("E5D_" + name, device=DEVICE)
        seg = load_annotation("annotation_" + name + "_iso_full")
        seg = seg[None, None, :, :y_crop, :] == 1
        self.img = self.img[None, None, :, :y_crop, :]

        self.mask_t = torch.tensor(seg, dtype=torch.bool, device=DEVICE)
        self.mask_f = ~self.mask_t
        self.nt = torch.sum(self.mask_t)

        # Logs
        self.steps = []
        self.log_f = []
        self.log_cost = []
        self.log_precision = []
        self.log_recall = []

    def cost(self, yp):
        global rpbal  # Where rpbal is defined as beta^2.
        
        tp = torch.sum(yp[self.mask_t])
        fp = torch.sum(yp[self.mask_f])
        fn = self.nt - tp
        if rpbal == 1.:
            fse = 1 - 2*tp / (2*tp + fn + fp)
        else:
            stp = (1 + rpbal)*tp
            fse = 1 - stp / (stp + rpbal*fn + fp)
        ypb = yp > 0.5
        tpb = torch.sum(ypb[self.mask_t])
        precision = tpb / torch.sum(ypb)
        recall = tpb / (self.nt * yp.shape[0])
        return fse, precision, recall

    def evaluate_output(self, model, training_step):
        img_y = model(self.img).cpu()
        out = (255 * img_y.detach().numpy().squeeze()).astype(np.uint8)
        nrrd.write(f"{model.name}_{training_step}_{self.name}.nrrd", out,
                   index_order="C", header=self.header)

    def log_step(self, step: int, cost, precision, recall):
        f_score = 2 / (1/precision + 1/recall)
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
    with open(f"logs/train_{model.name}_{TRAIN_VER}.pkl", "wb") as f:
        pickle.dump(training_logs, f)


# Due to how the volumes are batched, GPU training is not faster than CPU.
DEVICE = "cpu"
#DEVICE = torch.DEVICE("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on", DEVICE)
RAND_SEED = 43283
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED + 34921)
#torch.autograd.set_detect_anomaly(True)

CHKPT_DIR = "checkpoints"
TRAIN_VER = "dice"
VALIDATE_EVERY = 50  # Iterations
rpbal: float = 1.  # Adjusts how much recall is favored over precision, beta^2.

# Learning rate
LR_MIN = 1e-4
LR_MAX = 2e-3
LR_DECAY = 1/500  # Rate of learning decay per step.
LR_PERIOD = 400  # Number of steps per learning cycle.
lr_override = None


def main():
    global rpbal

    log_time = []
    log_lr = []
    log_val = []
    def log_everything():
        training_logs = {
            "time": log_time,
            "learning_rate": log_lr,
            "validation_f_score": log_val,
            "training_volumes": {tv.name: tv.logs_dict() for tv in training_volumes}
        }
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/train_{model.name}_{TRAIN_VER}.pkl", "wb") as f:
            pickle.dump(training_logs, f)

    model = MicroNet().to(DEVICE)
    print("Model:", model.name)
    training_volumes = np.array([
        TrainingVolume("082616_00_T46", 160, DEVICE),
        TrainingVolume("082616_01_T18", 160, DEVICE),
        TrainingVolume("082616_02_T45", 160, DEVICE),
        TrainingVolume("082616_04_T43", 150, DEVICE),
        TrainingVolume("082616_05_T16", 150, DEVICE),
        TrainingVolume("082616_06_T41", 150, DEVICE),
        TrainingVolume("082616_07_T15", 150, DEVICE),
        TrainingVolume("082616_08_T39", 160, DEVICE),
        TrainingVolume("082616_09_T14", 170, DEVICE),
        TrainingVolume("082616_10_T38", 170, DEVICE),
        TrainingVolume("082616_11_T14", 175, DEVICE),
        TrainingVolume("082616_12_T39", 190, DEVICE),
        TrainingVolume("082616_13_T14", 190, DEVICE),
    ], object)

    val = TrainingVolume("082616_03_T17", 170, DEVICE)
    test = TrainingVolume("082616_07_T40", 150, DEVICE)

    lr_override = None
    secs_cumulative = 0
    batch_iters = 3  # How many gradient descent iterations to make over each batch.
    step = 0  # Total number of gradient descent iterations completed.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MIN)
    os.makedirs(CHKPT_DIR, exist_ok=True)

    print("Enter command or training epochs.")
    while True:
        while True:
            r = input("> ")
            try:
                batch_iters = int(r)
                print(f"Looping {batch_iters} times over the dataset.")
                break
            except:
                pass
            if r.startswith("save "):
                ver = r.split()[1]
                torch.save(model.state_dict(),
                    f"{CHKPT_DIR}/{model.name}_{ver}.pt")
            elif r.startswith("load "):
                ver = r.split()[1]
                name = model.name + "_" + ver + ".pt"
                load_path = os.path.join(CHKPT_DIR, name)
                if not os.path.isfile(load_path):
                    load_path = os.path.join("trained_models", name)
                try:
                    model.load_state_dict(torch.load(load_path))
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
            elif r == "bootstrap":
                # A new zero layer that outputs four features.
                out4 = torch.zeros((4, 1, 3, 3, 3), dtype=torch.float32)
                mout = model.state_dict()
                mout["down1.1.weight"] = torch.cat((out4, mout["down1.1.weight"]), 1)
                mout["up0.0.weight"] = torch.cat((out4, mout["up0.0.weight"]), 1)
                torch.save(mout, "checkpoints/ur-net_bootstrap.pt")
            elif r == "val":
                val.evaluate_output(model, step)
            elif r == "test":
                test.evaluate_output(model, step)
            elif r == "finish":
                log_everything()
                return
            else:
                print("I didn't catch that.")

        for ie in range(batch_iters):
            batch_order = np.arange(len(training_volumes))
            np.random.shuffle(batch_order)
            for iv in batch_order:
                last_time = time.time()
                tv = training_volumes[iv]
                yp = model(tv.img)
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

                if (step + 1) % VALIDATE_EVERY == 0:
                    print("Validating... ", end="")
                    _, v_pre, v_rec = val.cost(model(val.img))
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
                           f"{CHKPT_DIR}/{model.name}_{TRAIN_VER}_{step}.pt")
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
        axes[0].plot(np.arange(len(log_val)) * VALIDATE_EVERY,
                     [-np.log10(1 - f) for f in log_val], "k.-")
        legend.append("Cross validation")
        #axes[0].set_ylim([0, 1])
        axes[0].set_title("-log10(1 - F_Score)")
        axes[1].legend(legend)
        axes[1].set_title("Cost")
        for a in axes:
            a.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
