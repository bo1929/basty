import logging
import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy.stats import zscore
from scipy.ndimage import median_filter as medfilt
from filterpy.kalman import KalmanFilter
from scipy.interpolate import splev, splrep

# This functions are only for coordinate values.
# Methods should be applied to pose dataframe directly.
# Filters are applied for each cartesian compenent of each body-part.


def rts_smoother_one_dimensional(df_pose, kalman_filter_kwargs, dim_x=2, dim_z=1):
    fk = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    # (dim_x, dim_x) process uncertainty, default eye(dim_x)
    fk.Q = kalman_filter_kwargs.get("process_uncertainty", np.eye(dim_x))
    # (dim_x, dim_x) state transistion matrix
    fk.F = kalman_filter_kwargs.get("state_transmission_matrix", np.eye(dim_x))
    # (dim_z, dim_x) measurement function
    fk.H = kalman_filter_kwargs.get("measurement_function", np.ones((dim_x, dim_z)))
    # (dim_x, dim_x) covariance matrix, default eye(dim_x)
    fk.P = kalman_filter_kwargs.get("covariance_matrix", np.eye(dim_x))
    # (dim_z, dim_z) measurement noise, default eye(dim_x)
    fk.R = kalman_filter_kwargs.get("noise", np.eye(dim_z))

    batch_size = kalman_filter_kwargs.get("batch_size", -1)
    if batch_size < 2:
        batch_size = df_pose.shape[0]
    t = np.arange(0, df_pose.shape[0])

    for col in df_pose.columns:
        pos = df_pose[col].to_numpy()
        f = splrep(t, pos, k=5, s=3)
        vel = splev(t, f, der=1) / 10
        for i in range(0, pos.shape[0], batch_size):
            # (dim_x, 1) initial location and velocity
            fk.x = np.array([[pos[i]], [vel[i]]])
            mu, cov, _, _ = fk.batch_filter(pos[i : i + batch_size])
            M, _, _, _ = fk.rts_smoother(mu, cov)
            pos[i : i + batch_size] = np.squeeze(M[:, 0])
        df_pose[col] = pos
    return df_pose


def rts_smoother_two_dimensional(df_pose):
    raise NotImplementedError


def boxcar_center_filter(df_pose, winsize):
    for name in df_pose.columns:
        df_pose[name] = (
            df_pose[name].rolling(window=winsize, min_periods=1, center=True).mean()
        )
    return df_pose


def median_filter(df_pose, winsize):
    for name in df_pose.columns:
        df_pose[name] = medfilt(df_pose[name].to_numpy(), size=winsize)
    return df_pose


def decimate(df_pose, q, ftype="iir"):
    decimated_dict = {}
    for name in df_pose.columns:
        decimated_dict[name] = signal.decimate(df_pose[name].to_numpy(), q, ftype=ftype)
    return pd.DataFrame.from_dict(decimated_dict)


def remove_decrasing_llh_frames(df_pose, df_llh, winsize=45, lower_z=-3):
    assert lower_z < 0
    pfilt = {}
    if winsize % 2 == 0:
        winsize += 1
    for col in df_llh.columns:
        llh = df_llh[col].astype(float).to_numpy()
        grad_llh = signal.savgol_filter(
            llh, window_length=winsize, polyorder=2, deriv=1
        )
        z = zscore(grad_llh, axis=0, ddof=0, nan_policy="omit")

        decreasing_idx = np.nonzero(z < lower_z)[0]
        pfilt[col] = round(decreasing_idx.shape[0] / llh.shape[0], 5)

        df_pose[col + "_x"].iloc[decreasing_idx] = np.nan
        df_pose[col + "_y"].iloc[decreasing_idx] = np.nan

    return (df_pose, pfilt)


def remove_low_llh_frames(df_pose, df_llh, threshold=0.15):
    pfilt = {}
    for col in df_llh.columns:
        llh = df_llh[col].astype(float)

        low_idx = np.nonzero(llh.to_numpy() < threshold)[0]
        pfilt[col] = round(low_idx.shape[0] / llh.shape[0], 5)

        df_pose[col + "_x"].iloc[low_idx] = np.nan
        df_pose[col + "_y"].iloc[low_idx] = np.nan

    return (df_pose, pfilt)


# Mark jumps as NaN based on the gradient quantiles.
def remove_jumps(df_pose, q=0.999):
    grad = np.gradient(df_pose.to_numpy(), axis=0, edge_order=2)
    quan_grad = np.quantile(grad, q, axis=0)
    columns = df_pose.columns
    num_interpolated = 0

    for i in range(quan_grad.shape[0]):
        idx_nan = np.nonzero(grad[:, i] > quan_grad[i])[0]
        num_interpolated += len(idx_nan)
        df_pose[columns[i]].iloc[idx_nan] = np.nan

    avgnum_interpolated = round(num_interpolated / quan_grad.shape[0], 2)
    logger = logging.getLogger("main")
    logger.info(f"avg. number of frames removed as jump: {avgnum_interpolated}.")
    return df_pose, avgnum_interpolated


# This is for fun, terribly slow and unnecessary.
def iteratively_impute_jumps(df_pose, q=0.999):
    df_pose, _ = remove_jumps(df_pose, q)

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(random_state=0)
    df_pose = imputer.fit_transform(df_pose.to_numpy())
    return df_pose


def remove_local_outliers(df_pose, winsize=15, threshold=9):
    for name in df_pose.columns:
        medfilt_pose = medfilt(df_pose[name].to_numpy(), size=winsize)
        df_pose[name].iloc[np.abs(df_pose[name] - medfilt_pose) > winsize] = np.nan
    return df_pose


def fill_interpolate(df_pose):
    df_pose = df_pose.interpolate(method="ffill", axis=0)
    df_pose = df_pose.interpolate(method="bfill", axis=0)
    df_pose = df_pose.interpolate(method="linear", axis=0)
    return df_pose


def forward_impute_jumps(df_pose):
    raise NotImplementedError
