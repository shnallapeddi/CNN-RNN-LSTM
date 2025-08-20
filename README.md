
## CNN · RNN · LSTM
A notebook suite exploring convolutional and recurrent sequence models, with emphasis on training dynamics (e.g., vanishing gradients), transfer learning, and baseline construction for text and time-series tasks.

### What this repo covers
#### CNN track (vision)

1. VGG → ResNet → ResNeXt progression: compares plain deep stacks vs. residual learning and group convolutions (cardinality).
2. Blocks & ops: conv→BN→nonlinearity, residual skip connections, bottlenecks (1×1–3×3–1×1), and grouped convs (ResNeXt).
3. Optimization: typical image-classification loop with cross-entropy loss, weight decay, and LR scheduling; shows how residuals stabilize training at depth.
4. Transfer learning: freezing backbones, discriminative learning rates for new heads, staged unfreezing, and light data augmentation.

#### RNN/LSTM track (sequences)

1. Vanishing/exploding gradients: empirical gradient-norm tracking across time; effects of activations, initialization, sequence length, and gradient clipping; how gating (LSTM) mitigates decay.
2. Sentiment with LSTM: tokenization → integerization → embeddings (random or pretrained) → (bi)LSTM → dense head; masking/padding for variable length; dropout regularization; accuracy/F1 evaluation.
3. Time-series with RNNs: sliding/rolling windows, teacher forcing vs. free-running forecasting, multi-step horizon evaluation (MAE/RMSE/MAPE); comparison to naïve/baseline predictors.

#### How to run
##### clone
git clone https://github.com/shnallapeddi/CNN-RNN-LSTM
cd CNN-RNN-LSTM

##### environment (choose the stack you prefer for the notebooks you open)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install jupyter numpy pandas matplotlib scikit-learn

##### if using PyTorch-based notebooks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu-only

##### if using Keras/TensorFlow-based notebooks
pip install tensorflow

#### Technical details worth noting

a. Gradient diagnostics: utilities to log per-time-step gradient norms and visualize decay/explosion; optional clip_grad_norm_ to stabilize long sequences.
b. Regularization: dropout in embeddings/recurrent layers, label smoothing (for classification heads), weight decay via optimizer.
c. Initialization: comparisons of Xavier/He with/without residual paths.
d. Scheduling: step or cosine annealing learning-rate schedules; early stopping callbacks.
e. Evaluation: classification → accuracy, precision/recall/F1; forecasting → MAE, RMSE, MAPE; plus residual/error distribution plots.
f. Reproducibility: seed setting and deterministic flags where supported

#### Suggested runs
a. Understand depth vs. trainability: run VGG to ResNet.ipynb, push depth upward, observe loss/grad stability with and without residuals.
b. See cardinality effects: run ResNeXt.ipynb and vary group counts to trade FLOPs vs. accuracy.
c. Feel vanishing gradients: run Investigating the Vanishing Gradient Problem.ipynb, plot gradient norms across time, then flip on gradient clipping and LSTM gating.
d. Baseline NLP: run Sentiment Analysis using LSTM.ipynb, start with a small embedding/LSTM, then toggle bidirectionality and dropout to see generalization change.
e. Forecasting: run Time Series Forecasting using RNNs.ipynb with your own CSV, tune window size and horizon; compare teacher-forced vs. open-loop predictions.
f. Fine-tune efficiently: run Transfer Learning with Pre-trained Models.ipynb, freeze backbone layers, train head, then unfreeze top blocks with a lower LR.





