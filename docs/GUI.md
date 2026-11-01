# BaseNILM GUI — User Guide

The BaseNILM GUI is a local web application (Streamlit) around the unchanged toolkit. All parameters of `start.py`
(`setupExp`, `setupDat`, `setupPar`) and `mdlPara.py` (`setupMdl`) are edited in the browser and stored as JSON
configs; the datasets are read with the loaders of the toolkit. The GUI runs on your own machine only
(`http://localhost:8501`), no data leave it.

## 1. Installation and start

```
pip install -r requirements.txt          # toolkit
pip install -r gui/requirements.txt      # GUI: streamlit, plotly, openpyxl
python start_gui.py                      # opens http://localhost:8501
```

`gui/requirements.txt` pins Streamlit < 1.61, protobuf < 5 and numpy < 2: newer Streamlit versions need protobuf ≥ 5,
which TensorFlow 2.16 does not accept. Install both files into the same environment and in this order.

Stop the GUI with `Ctrl+C` in the terminal. Server settings (port, localhost only, theme) are in
`.streamlit/config.toml`. The toolkit stays usable without the GUI: `python start.py` works as before.

## 2. Layout

| Area | Content |
|---|---|
| **Sidebar** | Expert mode, configuration manager (load / save / download / import / reset), runs of this session |
| **Setup** | Simulation name, author, output switches and the dataset (folder, format, training/testing files, appliances, inputs, limit, sampling) |
| **Input data** | Data explorer of every dataset of the selected folder |
| **0 · Simulation** | Training and/or testing (1-fold, k-fold, transfer, id based) |
| **1 · Hyperparameter optimisation** | Keras-Tuner Hyperband (TF) / GridSearchCV (SK) via `optiHyp.py` |
| **2 · Grid optimisation** | Simulations over a grid of one or two freely chosen parameters (e.g. window × fs) |
| **Results browser** | All folders of `results/`, incl. runs started with the start scripts |
| **Help** | Tutorials in three sections: *Quick start* (pretrained default DNN, CNN and LSTM, training off — results in under a minute), *Toolkit* (the files of `setup/tutorials`) and *Optimisation* (four tutorials), *Theory of the methods*, this guide |

Every mode page has the views **Workflow** (flow chart of the pipeline with the current settings, skipped steps
dashed), **Parameters** (tabs Experiment / Data / General / Model / Features, grid mode also *Grid*), **Inputs** (data
explorer), **Model** (network architecture, simulation and grid), **Theory** (all modes), **Run** and **Results**.

The **Theory** view of the optimisation modes explains the method with interactive toy examples: *1 · Hyperparameter
optimisation* — nested training / validation problem, grid vs random search, the Hyperband schedule of Keras-Tuner
(brackets, trials and epochs computed for the configured *Epochs*, successive halving on learning curves) and the
search space of the toolkit (TF hypermodel or the GridSearchCV grid of the selected SK model); *2 · Grid
optimisation* — cost of a full factorial grid, the time span W / fs of the window × sampling grid, full grid vs
one-at-a-time search with interacting parameters and run-to-run noise, and how to read the results.

**Help › Theory of the methods** explains the model families with interactive toy examples: neuron, activation
functions, gradient descent, dense network, convolution, LSTM, attention; random forest, KNN, SVM; DTW and NMF / DSC.

## 3. Parameters and configurations

The parameter list is generated from the Python sources: `gui/build_schema.py` reads every
`setupX['var'] = value  # comment` line of `start.py` and of `mdlPara()` — the banner titles give the categories,
the comments the help and the options (`0) cpu, 1) gpu`). `start_gui.py` rebuilds the schema when `start.py`,
`mdlPara.py` or `gui/adapter.py` changed; manually: `python gui/build_schema.py`. New parameters in `start.py` therefore
appear in the GUI automatically.

* **Basic / expert mode**: without expert mode only the main parameters that have an effect with the current settings
  are shown (e.g. *Folds* only for k-fold, the SK parameters only for the solver SK). Hidden parameters keep their
  values and are used.
* The model parameters are grouped by solver family (*Sklearn*, *Pattern matching*, *Source separation*) with a
  sub-heading per model (RF, SVM, KNN, DTW, …); in basic mode only the sub-group of the selected model is shown.
* The feature lists of `mdlPara.py` (statistical, rolling, 2D) have their own tab *Features*; they are used when
  *Features* (General › Framing and features) is switched on.
* **Configs**: *Save* writes `config/<name>.json`, *Load* reads it. `build_schema.py` also writes one ready-to-load
  config per start file of `setup/default`, `setup/tutorials`, `setup/benchmark` and `setup/journal`
  (`config/<folder>_<file>.json`, with the `mdlPara*.py` the file imports).
* **Import**: a JSON config or any BaseNILM start file (`.py`), which is parsed the same way.
* For a run the configuration is split into the four dicts and passed to `main()` / `optiHyp()` / `optiGrid()`
  directly; the GUI writes no files into the toolkit.

## 4. Data explorer

The dataset is read with the loader of its format (`loadMat`, `loadPkl`, `loadXlsx`, `loadH5`, `loadParquet` from
`src/data/loadData.py`) and cached. Files above 150 MB (e.g. `data/bus28/*.parquet`) are loaded only on request.

| Tab | Content |
|---|---|
| Overview | Samples, duration, sampling time, inputs, appliances (selected / total), ids, channel table with units, energy share and ON time |
| Time series | Aggregate input with the selected appliances (stacked), start/span selection |
| Appliances | Energy share (selected appliances highlighted), ON time above the threshold, power distribution, correlation inputs ↔ appliances |
| Split | Training / validation / testing along the time axis as `loadData()` selects them (1-fold, k-fold, transfer, ids) |
| Pre-processing & framing | Resampling (`resampleData`); the selected frame as a box on the series (one row per input feature, the next frames lighter), the frame itself and its target (`createSeq`: seq2point focus, seq2subseq / seq2seq range); optionally normalised as the model sees it (`normData`) |

Validation data: with *Validation split* (`valSplit`) = 0 (default, as before) the validation data are the first *rV*
share of the training part and also used for training; with 1 the last *rV* share (or the ids *idV*) is held out.

## 5. Model architecture

The view **Model** shows the network the toolkit builds for the current configuration. The GUI runs the training
pipeline of the toolkit (`main` → `train` → `trainMdlTF`) on the first 20,000 samples and stops it right before
`fit()`: the model comes from `src/model/models.py` with exactly the input and output shapes of a real run, nothing is
trained and nothing is written (the probe folder is removed). Shown are a layer diagram (convolutions blue, dense
green, recurrent orange, attention / custom blocks violet), a table of layers, output shapes and parameters, the
training setup and whether trained weights `mdl/mdl_<model>_<name>.weights.h5` exist (needed when *Training* is off).
The architecture is read-only — the layers are defined in `models.py`. SK / PM / SS solvers show their parameters.

## 6. Running and results

The **Run** view lists the settings and warnings (missing trained model, long k-fold, grid or full-data trainings)
and starts the run in the GUI process; the console output is streamed.

**Quick check** (switch in the Run view): 1 epoch, at most 20,000 samples, k-fold ≤ 2 folds, grid ≤ 2 × 2 points.
Trained models and the results folder get the suffix `_check`, so no trained model is overwritten — use it to test a
configuration before the real run; its scores are not meaningful.

The GUI records the results from the calls in `src/test.py` (`performance`, `featureRank`, `printResults`) — the
numbers are identical to the console table of `start.py`. **Results** shows:

* tiles of the AVG row (TECA, F1, accuracy, RMSE, MAE) and the per-appliance table (same columns as the console);
* scores per appliance, ground truth vs. prediction over time, scatter and error distribution;
* estimated vs. true energy shares and the estimated energy breakdown;
* training curves (`training_log.csv`) and the feature ranking;
* k-fold: fold selector, mean ± standard deviation, box plots across the folds;
* hyperparameter optimisation: best hyperparameters, score per trial, parallel coordinates of all trials, final
  training curve; SK: *Apply to the configuration* (TF: the learning rate);
* grid optimisation: heatmap (two variables) or line (one variable) of the selected score — average or one appliance
  — with the best point and *Apply best values*, score per appliance and grid point, run time per point;
* the matplotlib figures of `plotting.py` (*Figures & files*) and the console log.

Runs are stored as `gui_run.pkl` / `gui_run.json` in `results/<name>` (created if *Save* is off) and can be re-opened
in the results browser. Long time series are thinned to 20,000 points for storage.

## 7. Grid and hyperparameter optimisation

* **Grid** (Parameters › Grid): variable 1 and 2 are any numeric or option parameter (window, fs, learning rate,
  epochs, batch, normalisation, SK parameters, …), variable 2 may be *none* (1-D sweep). Values are comma-separated
  (fractions such as `1/60` allowed). *Couple framing to the window* sets overlap = W − 1 and focus = W / 2 at every
  point as `optiGrid.py` does. Each point is a complete `main()` run named `<name>_G<i>_<j>`; its model is deleted
  afterwards unless *Keep the models* is on. `optiGrid.py` (window × fs) is still used by `start.py` with `sim = 2`.
* **Hyperparameters**: `optiHyp.py` writes `results/<name>/opti_<model>_<name>.json` (best values, all trials,
  final training history), which the GUI shows.

## 8. Changes in the toolkit (November 2026)

| Topic | Change |
|---|---|
| Split | X and y are split with the same indices (with *Shuffle* on they were split independently); option `valSplit` |
| `trainBatch = 2` | trains per id (`tempData`), it trained on all data per id before |
| Loaders | clear errors with the reason, `csv` = `<name>_input.csv`, `_output.csv`, `_units.csv`, sub-folder fallback, H5 needs nilmtk |
| Testing | stops with an error if the trained model is missing (it predicted with an untrained model) |
| GPU | `gpu = 1` without GPU falls back to the CPU; the PT models follow `gpu` (they used CUDA whenever available and tested on the CPU), predict in batches on the selected device |
| Normalisation | quantile output normalisation (5) transforms y (it used X); mean/std (3) divides by the standard deviation (it divided by the variance); Q1/Q3 (4) takes the quantiles of the right data — models trained before with options 3/4 must be retrained |
| SK training | the model is always built from the current parameters (an existing model file was loaded and refitted, so changed parameters were ignored) |
| KNN | new parameter `SK_KNN_weights` (uniform / distance, default distance as before); the loop always used 'distance' |
| TECA | training metric and per-appliance TECA use Σ\|y\| |
| Hyperparameters | Keras 3 metrics lists, output size of the tuned models, MultiOutput grid keys, best epoch = min val_loss, JSON result |
| Resampling | downsampling averages each interval (zero-order hold dropped short events); default fs = raw fs is unchanged |
| Others | `save.py` (paths, grid run time, k-fold folder index), `featureRank` with `outEnergy`, model size, messages |

The default `start.py` run normalises its inputs with option 3, so the shipped model `mdl/mdl_CNN_Default` (trained with the old scaling) must be retrained after the normalisation fix.

## 9. Troubleshooting

| Problem | Remedy |
|---|---|
| Port 8501 in use | Stop the other instance or change `server.port` in `.streamlit/config.toml` |
| Changes in the code are not visible | Restart `start_gui.py` |
| `numpy` / `protobuf` conflicts after installing | Re-install `gui/requirements.txt` (it pins versions TensorFlow 2.16 accepts) |
| The appliance list is a text field | The dataset is not loaded (large file or wrong folder/format) — open *Inputs* |
| Model view fails | The pipeline stopped before `fit()`; the log of the toolkit is shown — usually a dataset / shape problem |
| *No GPU found, running on CPU* with `gpu = 1` (TF) | TensorFlow ≥ 2.11 has no GPU support on native Windows (2.16 is a CPU build); use WSL2 with `pip install tensorflow[and-cuda]` |
| *PyTorch finds no GPU* with `gpu = 1` (PT) | Install the CUDA build: `pip install --force-reinstall --no-deps torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121` |
