"""
Training checklist I actually follow in practice:
0. Fix seeds so results are reproducible
1. Overfit one sample (debug gradients early)
2. Clean data pipeline (no leakage)
3. Establish a dumb baseline
4. Overfit a small subset to check capacity
5. Add regularization only after it works
6. Tune learning rate + decay
7. Use mini-batches
8. Track losses, accuracy, gradients
9. Touch test set once at the end
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL SEED  
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import os, time, json, warnings
warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
CFG = {
    "seed"        : GLOBAL_SEED,
    "dataset"     : "sklearn_digits",  
    "test_size"   : 0.15,
    "val_size"    : 0.15,
    "hidden_dims" : [256, 128, 64],    
    "activation"  : "relu",
    "lr"          : 0.01,
    "lr_decay"    : 0.95,              
    "lr_decay_every": 20,
    "batch_size"  : 32,
    "epochs"      : 150,
    "l2_lambda"   : 1e-4,          
    "dropout_rate": 0.3,              
    "output_dir"  : "/mnt/user-data/outputs/nn_golden_rules",
}

os.makedirs(CFG["output_dir"], exist_ok=True)
print("=" * 70)
print("  GOLDEN RULES EXPERIMENT —  Neural Network from Scratch")
print("=" * 70)
print(f"\n[Config] Global seed : {CFG['seed']}")
print(f"[Config] Architecture: 64 → {CFG['hidden_dims']} → 10")
print(f"[Config] Learning rate: {CFG['lr']}  |  Batch: {CFG['batch_size']}  |  Epochs: {CFG['epochs']}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET LOADING & PRE-PROCESSING  ← Rule 2: Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 1 — Loading & Pre-processing Data")
print("─" * 70)

digits = load_digits()
X_raw, y_raw = digits.data, digits.target  # (1797, 64), (1797,)

print(f"\n[Data] Total samples : {X_raw.shape[0]}")
print(f"[Data] Feature dims  : {X_raw.shape[1]}  (8×8 pixel intensities)")
print(f"[Data] Classes       : {len(np.unique(y_raw))}  (digits 0–9)")
print(f"[Data] Class dist    : {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# ── First split off test set 
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_raw,
    test_size=CFG["test_size"],
    random_state=CFG["seed"],
    stratify=y_raw   
)

# ── Split remaining into train / validation ──
val_fraction = CFG["val_size"] / (1 - CFG["test_size"])
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_fraction,
    random_state=CFG["seed"],
    stratify=y_temp
)

print(f"\n[Split] Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# Fit scaler on train only — never leak val/test stats
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_val   = scaler.transform(X_val)         
X_test  = scaler.transform(X_test)        

print(f"[Norm] Train mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f}  (should be ~0, ~1)")
print(f"[Norm] Val   mean≈{X_val.mean():.4f}, std≈{X_val.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BASELINE — Dummy Classifier 
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 2 — Baseline: Dummy Classifier (Most-Frequent Strategy)")
print("─" * 70)
# If NN can't beat random guessing, something is deeply wrong
dummy = DummyClassifier(strategy="most_frequent", random_state=CFG["seed"])
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_val, dummy.predict(X_val))
print(f"\n[Baseline] Dummy accuracy on val set : {dummy_acc:.4f}  ({dummy_acc*100:.1f}%)")
print(f"[Baseline] Our NN must significantly exceed {dummy_acc*100:.1f}% to be worth anything.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEURAL NETWORK — Pure NumPy Implementation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 3 — Building the Neural Network (No Framework — Pure Math)")
print("─" * 70)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true, weights=None, l2_lambda=0.0):
    """
    Cross-entropy loss with  L2 regularisation to penalises large weights, reduces overfitting.
    """
    n = len(y_true)
    probs_clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    ce = -np.mean(np.log(probs_clipped[np.arange(n), y_true]))
    l2 = 0.0
    if weights and l2_lambda > 0:
        l2 = (l2_lambda / 2) * sum(np.sum(w ** 2) for w in weights)
    return ce + l2


class NeuralNetwork:
    """
    Simple MLP implemented from scratch (NumPy only).
    Supports ReLU, dropout, L2 regularization, and SGD.
    """

    def __init__(self, layer_dims, l2_lambda=1e-4, dropout_rate=0.3, seed=42):
        np.random.seed(seed)  
        self.layer_dims   = layer_dims    
        self.l2_lambda    = l2_lambda
        self.dropout_rate = dropout_rate
        self.n_layers     = len(layer_dims) - 1
        self.params       = {}             
        self.grad_norms   = []            

        # ── He Initialisation ──────────────────────────────────────────────
        # For ReLU: std = sqrt(2 / fan_in)
        # Too large → exploding gradients. Too small → vanishing gradients.
        for l in range(1, self.n_layers + 1):
            fan_in  = layer_dims[l - 1]
            fan_out = layer_dims[l]
            self.params[f"W{l}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{l}"] = np.zeros((1, fan_out))

        print(f"\n[NN Init] Layers: {layer_dims}")
        total_params = sum(
            self.params[f"W{l}"].size + self.params[f"b{l}"].size
            for l in range(1, self.n_layers + 1)
        )
        print(f"[NN Init] Total trainable parameters: {total_params:,}")
        print(f"[NN Init] L2 λ={l2_lambda}  |  Dropout={dropout_rate}")

    def forward(self, X, training=True):
        """
        Forward pass through all layers.
        Dropout only during training not at inference
        """
        cache = {"A0": X}
        A = X
        for l in range(1, self.n_layers + 1):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            Z = A @ W + b               
            cache[f"Z{l}"] = Z

            if l < self.n_layers:    
                A = relu(Z)
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                    A /= (1.0 - self.dropout_rate)
                    A *= mask
                    cache[f"mask{l}"] = mask
                else:
                    cache[f"mask{l}"] = np.ones_like(A)

            else:                         
                A = softmax(Z)

            cache[f"A{l}"] = A

        return A, cache

    def backward(self, cache, y_true):
        """
        Backpropagation — chain rule to compute gradients of the loss with respect to the weight
        """
        n = len(y_true)
        grads   = {}
        weights = [self.params[f"W{l}"] for l in range(1, self.n_layers + 1)]

        # Gradient of softmax + cross-entropy 
        dA = cache[f"A{self.n_layers}"].copy()
        dA[np.arange(n), y_true] -= 1
        dA /= n
        for l in range(self.n_layers, 0, -1):
            A_prev = cache[f"A{l-1}"]
            W      = self.params[f"W{l}"]
            grads[f"dW{l}"] = A_prev.T @ dA + self.l2_lambda * W
            grads[f"db{l}"] = np.sum(dA, axis=0, keepdims=True)

            if l > 1:
                dA = dA @ W.T
                mask = cache[f"mask{l-1}"]
                dA *= mask / (1.0 - self.dropout_rate + 1e-8)
                dA *= relu_derivative(cache[f"Z{l-1}"])

        return grads

    def update(self, grads, lr):
        """Vanilla SGD weight update: W ← W - lr * dW"""
        for l in range(1, self.n_layers + 1):
            self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= lr * grads[f"db{l}"]

    def compute_grad_norm(self, grads):
        """
       if norm bigger than 100 mean its exploding, if smaller than 1e-6 mean its vanishing
        """
        total = sum(np.sum(grads[k] ** 2) for k in grads if k.startswith("d"))
        return np.sqrt(total)

    def predict(self, X):
        probs, _ = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RULE 1 — SANITY CHECK: OVERFIT A SINGLE SAMPLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 4 — SANITY CHECK: Overfit on ONE Single Training Sample")
print("─" * 70)
print("""
# Quick sanity check:
# The model should easily overfit one sample.
# If this fails, gradients or loss are wrong.
""")

X_one = X_train[:1]   
y_one = y_train[:1]   
layer_dims = [X_train.shape[1]] + CFG["hidden_dims"] + [10]

sanity_net   = NeuralNetwork(layer_dims, l2_lambda=0.0, dropout_rate=0.0, seed=CFG["seed"])
sanity_losses = []
for epoch in range(300):
    probs, cache = sanity_net.forward(X_one, training=True)
    loss = cross_entropy_loss(probs, y_one)
    grads = sanity_net.backward(cache, y_one)
    sanity_net.update(grads, lr=0.05)
    sanity_losses.append(loss)

final_prob  = sanity_net.forward(X_one, training=False)[0]
final_pred  = np.argmax(final_prob)
sanity_acc  = int(final_pred == y_one[0]) * 100

print(f"[Sanity] True label  : {y_one[0]}")
print(f"[Sanity] Predicted   : {final_pred}")
print(f"[Sanity] Final loss  : {sanity_losses[-1]:.6f}  (should be near 0)")
print(f"[Sanity] Accuracy    : {sanity_acc}%  (should be 100%)")

if sanity_acc == 100 and sanity_losses[-1] < 0.01:
    print("[Sanity] ✅ PASSED — Network can overfit. Gradients flow correctly!")
else:
    print("[Sanity] ❌ FAILED — Check your forward/backward pass before proceeding!")



# ─────────────────────────────────────────────────────────────────────────────
# 5. RULE 4 — OVERFIT ON SMALL SUBSET (100 samples)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 5 — Overfit on 100-sample Subset (Capacity Check)")
print("─" * 70)
print("""
Before full training, verify the model has ENOUGH CAPACITY
to learn the task. If it can't overfit 100 samples, the model is too small
or the learning rate is wrong.
""")

np.random.seed(CFG["seed"])
idx_small   = np.random.choice(len(X_train), size=100, replace=False)
X_small     = X_train[idx_small]
y_small     = y_train[idx_small]

small_net    = NeuralNetwork(layer_dims, l2_lambda=0.0, dropout_rate=0.0, seed=CFG["seed"])
small_losses = []

for epoch in range(500):
    probs, cache = small_net.forward(X_small, training=False)  
    loss = cross_entropy_loss(probs, y_small)
    grads = small_net.backward(cache, y_small)
    small_net.update(grads, lr=0.05)
    small_losses.append(loss)

small_acc = small_net.accuracy(X_small, y_small)
print(f"[Capacity] Final loss on 100 samples : {small_losses[-1]:.4f}")
print(f"[Capacity] Train accuracy (100 samp) : {small_acc*100:.1f}%")
if small_acc > 0.97:
    print("[Capacity] ✅ Model has sufficient capacity to overfit the data.")
else:
    print("[Capacity] ⚠️  Model might need more capacity or lower LR.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL TRAINING WITH ALL GOLDEN RULES APPLIED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 6 — Full Training (All Golden Rules Active)")
print("─" * 70)
print(f"""
 Now we train on all {len(X_train)} training samples with all the rules applied.
""")

np.random.seed(CFG["seed"])
model = NeuralNetwork(
    layer_dims,
    l2_lambda    = CFG["l2_lambda"],
    dropout_rate = CFG["dropout_rate"],
    seed         = CFG["seed"]
)

train_losses, val_losses   = [], []
train_accs,   val_accs     = [], []
grad_norms                 = []
lr_history                 = []

best_val_acc   = 0.0
best_weights   = None   
patience       = 20     
no_improve     = 0
lr             = CFG["lr"]
n_train        = len(X_train)

start_time = time.time()

for epoch in range(1, CFG["epochs"] + 1):

    #  LR Decay 
    # Step decay: reduce LR every N epochs by a fixed factor.
    if epoch % CFG["lr_decay_every"] == 0:
        lr *= CFG["lr_decay"]

    lr_history.append(lr)

    # Mini-batch shuffle 
    # Shuffle every epoch to prevents the network from memorising sample order.
    idx = np.random.permutation(n_train)
    X_shuffled = X_train[idx]
    y_shuffled = y_train[idx]

    epoch_loss  = 0.0
    epoch_gnorm = 0.0
    n_batches   = 0

    for start in range(0, n_train, CFG["batch_size"]):
        end      = start + CFG["batch_size"]
        X_batch  = X_shuffled[start:end]
        y_batch  = y_shuffled[start:end]

        # Forward
        probs, cache = model.forward(X_batch, training=True)

        # Loss (with L2)
        weights = [model.params[f"W{l}"] for l in range(1, model.n_layers + 1)]
        batch_loss = cross_entropy_loss(probs, y_batch, weights, CFG["l2_lambda"])
        epoch_loss += batch_loss

        # Backward
        grads = model.backward(cache, y_batch)
        # Gradient norm — logged per batch, averaged per epoch
        epoch_gnorm += model.compute_grad_norm(grads)
        n_batches   += 1

        # Update
        model.update(grads, lr)

    # ── Per-epoch metrics ──────────────────────────────────────────────────
    avg_loss  = epoch_loss / n_batches
    avg_gnorm = epoch_gnorm / n_batches

    # Eval on full sets (no dropout at inference)
    train_acc = model.accuracy(X_train, y_train)
    val_acc   = model.accuracy(X_val, y_val)

    val_probs, _ = model.forward(X_val, training=False)
    val_loss = cross_entropy_loss(val_probs, y_val)

    train_losses.append(avg_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    grad_norms.append(avg_gnorm)

    # ── Model Checkpointing 
    # Save best weights based on validation accuracy 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        # Deep copy of weights
        best_weights = {k: v.copy() for k, v in model.params.items()}
        no_improve   = 0
    else:
        no_improve += 1

    # ── Early Stopping ─────────────────────────────────────────────────────

    if no_improve >= patience:
        print(f"\n[Early Stop] No val improvement for {patience} epochs. Stopping at epoch {epoch}.")
        break

    # ── Logging ───────────────────────────────────────────────────────────
    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:3d}/{CFG['epochs']} | "
              f"LR={lr:.5f} | "
              f"Loss={avg_loss:.4f} | "
              f"Val={val_acc*100:.1f}% | "
              f"Train={train_acc*100:.1f}% | "
              f"|∇|={avg_gnorm:.4f} | "
              f"t={elapsed:.1f}s")

print(f"\n[Train] Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
total_time = time.time() - start_time
print(f"[Train] Total training time: {total_time:.2f}s")

# ── Restore best weights ───────────────────────────────────────────────────
model.params = best_weights
print("[Train] Restored best model weights from epoch", best_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# 7. RULE 9 — FINAL EVALUATION ON TEST SET 
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 7 — Final Evaluation on HELD-OUT Test Set")
print("─" * 70)
print("""
# The test set is evaluated exactly once.
# No tuning, no debugging, no decisions are made based on test results.
# This provides an unbiased estimate of real-world performance.
""")

test_acc = model.accuracy(X_test, y_test)
y_pred   = model.predict(X_test)

print(f"[Final] Dummy Baseline : {dummy_acc*100:.2f}%")
print(f"[Final] Our NN Model   : {test_acc*100:.2f}%")
print(f"[Final] Improvement    : +{(test_acc - dummy_acc)*100:.2f}%")
print(f"\n[Report]\n{classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS — Rule 8: Monitor Everything
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 8 — Generating Diagnostic Plots")
print("─" * 70)

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0f0f1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

GOLD  = "#FFD700"
CYAN  = "#00FFCC"
CORAL = "#FF6B6B"
LIME  = "#90EE90"
GREY  = "#888888"
BG    = "#0f0f1a"
AX_BG = "#1a1a2e"

def style_ax(ax, title):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.set_title(title, color=GOLD, fontsize=10, fontweight="bold", pad=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

epochs_ran = len(train_losses)
ep_range   = range(1, epochs_ran + 1)

# ── Plot 1: Sanity Check Loss ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(sanity_losses, color=CYAN, linewidth=1.5)
ax1.axhline(0.01, color=CORAL, linestyle="--", linewidth=1, label="target < 0.01")
style_ax(ax1, "Rule 1 — Sanity Check (1 Sample)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.legend(fontsize=7, labelcolor=CORAL)
ax1.text(0.98, 0.85, f"Final: {sanity_losses[-1]:.4f}", transform=ax1.transAxes,
         color=LIME, fontsize=8, ha='right')

# ── Plot 2: Small-Set Overfitting ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(small_losses, color=GOLD, linewidth=1.5)
style_ax(ax2, "Rule 4 — Overfit 100 Samples")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Train Loss")
ax2.text(0.98, 0.85, f"Acc: {small_acc*100:.1f}%", transform=ax2.transAxes,
         color=LIME, fontsize=8, ha='right')

# ── Plot 3: Train vs Val Loss ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(ep_range, train_losses, color=CYAN,  linewidth=1.5, label="Train Loss")
ax3.plot(ep_range, val_losses,   color=CORAL, linewidth=1.5, label="Val Loss",   linestyle="--")
ax3.axvline(best_epoch, color=LIME, linestyle=":", linewidth=1.2, label=f"Best ep={best_epoch}")
style_ax(ax3, "Rule 8 — Train vs Val Loss")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.legend(fontsize=7, labelcolor="white")

# ── Plot 4: Train vs Val Accuracy ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(ep_range, [a*100 for a in train_accs], color=CYAN,  linewidth=1.5, label="Train Acc")
ax4.plot(ep_range, [a*100 for a in val_accs],   color=CORAL, linewidth=1.5, label="Val Acc", linestyle="--")
ax4.axhline(dummy_acc * 100, color=GREY, linestyle=":", linewidth=1, label=f"Dummy {dummy_acc*100:.0f}%")
ax4.axvline(best_epoch, color=LIME, linestyle=":", linewidth=1.2)
style_ax(ax4, "Rule 8 — Accuracy Curves")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy (%)")
ax4.legend(fontsize=7, labelcolor="white")

# ── Plot 5: Gradient Norm ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(ep_range, grad_norms, color=GOLD, linewidth=1.2)
ax5.axhline(100, color=CORAL, linestyle="--", linewidth=1, label="Explosion risk")
ax5.axhline(1e-4, color=CYAN, linestyle="--", linewidth=1, label="Vanishing risk")
style_ax(ax5, "Rule 8 — Gradient Norm ‖∇‖")
ax5.set_xlabel("Epoch")
ax5.set_ylabel("|Gradient|")
ax5.set_yscale("log")
ax5.legend(fontsize=7, labelcolor="white")

# ── Plot 6: Learning Rate Schedule ───────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(ep_range, lr_history, color=LIME, linewidth=1.5)
style_ax(ax6, "Rule 6 — LR Schedule (Step Decay)")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Learning Rate")
ax6.set_yscale("log")

# ── Plot 7: Weight Distribution (Layer 1) ────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
w1 = model.params["W1"].flatten()
ax7.hist(w1, bins=50, color=CYAN, edgecolor="none", alpha=0.8)
ax7.axvline(w1.mean(), color=GOLD, linestyle="--", linewidth=1.5, label=f"μ={w1.mean():.3f}")
style_ax(ax7, "Rule 8 — W1 Weight Distribution")
ax7.set_xlabel("Weight Value")
ax7.set_ylabel("Count")
ax7.legend(fontsize=7, labelcolor="white")

# ── Plot 8: Sample digit predictions ─────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(AX_BG)
ax8.axis("off")
ax8.set_title("Rule 9 — Test Samples vs Predictions", color=GOLD, fontsize=10, fontweight="bold")

# Show 8 test samples in a 2×4 grid within this subplot
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[2, 1], hspace=0.1, wspace=0.1)
for i in range(8):
    inner_ax = fig.add_subplot(inner_gs[i // 4, i % 4])
    inner_ax.set_facecolor(AX_BG)
    img = scaler.inverse_transform(X_test[i:i+1]).reshape(8, 8)
    pred = model.predict(X_test[i:i+1])[0]
    true = y_test[i]
    color = LIME if pred == true else CORAL
    inner_ax.imshow(img, cmap="hot", interpolation="nearest")
    inner_ax.set_title(f"P:{pred}", color=color, fontsize=7, pad=1)
    inner_ax.axis("off")

# ── Plot 9: Final Summary Stats 
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(AX_BG)
ax9.axis("off")
style_ax(ax9, "Experiment Summary")

summary_lines = [
    ("Dataset",         "sklearn Digits"),
    ("Samples (train)", f"{len(X_train)}"),
    ("Architecture",    f"64→256→128→64→10"),
    ("Total Params",    f"{sum(v.size for k,v in model.params.items() if 'W' in k or 'b' in k):,}"),
    ("", ""),
    ("Dummy Baseline",  f"{dummy_acc*100:.1f}%"),
    ("Val Accuracy",    f"{best_val_acc*100:.2f}%  (ep {best_epoch})"),
    ("TEST Accuracy",   f"{test_acc*100:.2f}%  ← FINAL"),
    ("", ""),
    ("L2 Lambda",       f"{CFG['l2_lambda']}"),
    ("Dropout",         f"{CFG['dropout_rate']}"),
    ("Batch Size",      f"{CFG['batch_size']}"),
    ("Init LR",         f"{CFG['lr']}"),
    ("Train time",      f"{total_time:.1f}s"),
]

y_pos = 0.95
for label, value in summary_lines:
    if not label:
        y_pos -= 0.04
        continue
    ax9.text(0.02, y_pos, f"{label}:", color=GREY,  fontsize=8, transform=ax9.transAxes)
    ax9.text(0.55, y_pos, value,       color=CYAN,  fontsize=8, transform=ax9.transAxes,
             fontweight="bold" if "TEST" in label else "normal")
    y_pos -= 0.065

fig.suptitle(
    "The Golden Rules of Neural Network Training & Optimization\n"
    "NumPy from Scratch  |  sklearn Digits Dataset",
    color=GOLD, fontsize=14, fontweight="bold", y=0.98
)

plot_path = os.path.join(CFG["output_dir"], "golden_rules_diagnostics.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[Plot] Diagnostic plot saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. REPRODUCIBILITY REPORT — Save config + results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "config"          : CFG,
    "sanity_check"    : {"final_loss": float(sanity_losses[-1]), "accuracy": sanity_acc},
    "capacity_check"  : {"final_loss": float(small_losses[-1]), "accuracy": float(small_acc)},
    "training"        : {
        "best_epoch"    : int(best_epoch),
        "best_val_acc"  : float(best_val_acc),
        "train_time_s"  : float(total_time),
    },
    "final_test"      : {
        "dummy_baseline": float(dummy_acc),
        "nn_accuracy"   : float(test_acc),
        "improvement"   : float(test_acc - dummy_acc),
    }
}

results_path = os.path.join(CFG["output_dir"], "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"[Results] Saved → {results_path}")
print("\n" + "=" * 70)
print("  EXPERIMENT COMPLETE")
print("=" * 70)
print(f"  Sanity Check  : {'✅ PASSED' if sanity_acc == 100 else '❌ FAILED'}")
print(f"  Capacity Check: {'✅ PASSED' if small_acc > 0.97 else '⚠️  PARTIAL'}")
print(f"  Dummy Baseline: {dummy_acc*100:.1f}%")
print(f"  Final Test Acc: {test_acc*100:.2f}%")
print(f"  Seed used     : {CFG['seed']}  — run again → same numbers!")
print("=" * 70)
