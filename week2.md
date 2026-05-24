# week2.md — instructions for Claude Code

## Project context

This is the Week 2 portion (Task 2.1) of the SF3 Machine Learning project
on cartpole modelling. The goal of Task 2.1 is to fit a nonlinear model
of the one-step cartpole dynamics using kernel regression with a
Gaussian basis and a periodic angle.

The Week-1 portion (linear `Y = C X` model) is in `linear_model.py` and
is used as a baseline / residual target.

## Conventions

- State vector ordering: `[x, x_dot, theta, theta_dot]`. Component 2 is
  the angle and is periodic on $(-\pi, \pi]$. Anywhere the angle
  appears in a difference, the periodic substitution `sin((a-b)/2)^2`
  applies in place of `(a-b)^2`.
- Never call `np.linalg.inv`. Use `np.linalg.lstsq` on the linear system
  for solving regularised normal equations.
- All randomness goes through `numpy.random.Generator` instances passed
  in explicitly. No global RNG state.
- Don't add new top-level dependencies. The project should run with
  numpy + matplotlib only. JAX is for Task 2.2, not Task 2.1.
- The `CartPole` class is provided and must not be edited.

## Workflow

The end-to-end pipeline is in `run_task_2_1.py`. Edit the `CFG` dict at
the top of that file to change the data size / hyperparameter grids /
rollout settings — don't sprinkle magic numbers into the helper modules.

To run:

```bash
python run_task_2_1.py
```

Outputs land in `outputs/<tag>/` where `<tag>` is one of
`{delta_NN, delta_NM, resid_NN, resid_NM}`. The `delta` tag uses the
direct change-in-state as the regression target; `resid` uses the
linear-model residual.

## What's done

- Data gathering and splitting (`data.py`).
- Linear baseline (`linear_model.py`).
- Periodic-angle Gaussian kernel (`kernel.py`).
- Both regression forms ($N\!\times\!N$ Tikhonov and sparse $N\!\times\!M$
  with kernel-norm regularisation) in `nonlinear_model.py`.
- Lambda sweep on validation set.
- Convergence sweeps over $N$ and over $M$.
- Pred-vs-true scatter, 1D scans, 2D slice contours, iterated rollouts.
- `results.json` collecting every quantitative number.

## What's NOT done (handle in later tasks)

- Task 2.2 (gradient-based hyperparameter tuning with JAX) — add a new
  module `hyper.py`. Don't break the existing numpy fitting code; rather
  add JAX-equivalent versions alongside.
- Task 2.3 (`sin/cos` features) — when you implement this, extend the
  state to a 5-vector internally and drop the periodic-angle kernel
  modification. Add a clear feature-lifting function in `kernel.py`
  that the regression code can call.
- Task 3+ (control) — actions get added to the state, max_force may
  need re-tuning.

## Sanity checks before declaring "done"

1. `python run_task_2_1.py` completes without errors.
2. `outputs/results.json` exists and contains a `linear_test_mse` near 1.0
   and `runs.delta_NN.headline_test_mse` ~ two orders of magnitude lower.
3. `outputs/delta_NM/06_convergence_M.png` shows monotonic decrease as
   $M$ doubles (modulo small fluctuations at the smallest $M$).
4. `outputs/delta_NN/01_pred_vs_true.png` shows points hugging the
   diagonal across all four components.
5. `outputs/delta_NN/04_rollouts.png` shows the orange model curve
   tracking the black true curve closely for several seconds on the
   near-equilibrium initial conditions.