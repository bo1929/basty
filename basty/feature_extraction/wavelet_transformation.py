import logging
import math
from copy import deepcopy

import numpy as np
import pywt


class WaveletTransformation:
    def __init__(self, fps, wv_cfg={}):
        self.wv_cfg = deepcopy(wv_cfg)
        self.logger = logging.getLogger("main")

        assert fps > 0
        self.sampling_freq = fps

        self.wavelet = self.wv_cfg.get("wavelet", "morl")
        self.padding = self.wv_cfg.get("padding", "reflect")
        self.num_channels = self.wv_cfg.get("num_channels", 20)
        self.freq_low = self.wv_cfg.get("freq_low", 0.5)
        self.freq_up = self.wv_cfg.get("freq_up", self.sampling_freq / 2)
        self.freq_spacing = self.wv_cfg.get("freq_spacing", "dyadic")
        self.normalization_option = self.wv_cfg.get("normalization_option", 2)

        assert self.num_channels > 1
        assert self.freq_low > 0
        assert self.freq_spacing in ["dyadic", "linear"]

        if self.freq_up > (self.sampling_freq / 2):
            self.logger.warning(
                f"Upper bound frequency {self.freq_up} is greater than Nyquist frequency."
            )
            self.logger.info(
                f"Upper frequency is set to {round(self.sampling_freq/2, 3)}."
            )
            self.freq_up = self.sampling_freq / 2

        self.w0 = 5
        self.dc = 0.8125
        # In terms of Hz.
        self.dt = 1 / self.sampling_freq
        self.scales, self.freq_range = self._get_scales()

    def _get_scales(self):
        if self.wavelet == "morl":
            if self.freq_spacing == "dyadic":
                exp_f = math.log2(self.freq_up / self.freq_low)
                freq_range = self.freq_up * np.exp2(
                    np.arange(0, self.num_channels) * exp_f / (-self.num_channels + 1)
                )
                scales = self.dc * 1 / (freq_range * self.dt)
                # This is also possible, precision is worse though.
                # scales = (self.w0 + math.sqrt(2+self.w0**2)) / (4*math.pi*freq_range*self.dt)
                # freq_range = pywt.scale2frequency(self.wavelet, scales) / self.dt
            if self.freq_spacing == "linear":
                freq_range = np.linspace(self.freq_low, self.freq_up, self.num_channels)
                scales = self.dc * 1 / (freq_range * self.dt)
        else:
            raise NotImplementedError(
                "Given wavelet {self.wavelet} is not implemented yet."
            )
        return scales, freq_range

    def cwtransform(self, feature_array):
        time_shape = feature_array.shape[0]
        scale_shape = self.scales.shape[0]
        feature_shape = feature_array.shape[1]

        # Using entire time axis as pad-width might be slow.
        pad_width = time_shape
        cwt_coefs = np.ndarray((time_shape, scale_shape, feature_shape))

        for i in range(feature_shape):
            padded = pywt.pad(feature_array[:, i], pad_width, self.padding)
            cws = pywt.cwt(padded, self.scales, self.wavelet, self.dt)
            ws = cws[0][:, time_shape : time_shape + pad_width]
            cwt_coefs[:, :, i] = np.swapaxes(ws, 0, 1)

        return cwt_coefs

    def normalize_channels(self, cwt_coefs):
        """

        Normalizes each time scale according to given option.

        normalization_option == 0
            pass
        normalization_option == 1
            Berman, Choi, Bialek, Shaevitz, (2014),
            Mapping the stereotyped behaviour of freely moving fruit flies,
            Journal of The Royal Society Interface.
        normalization_option == 2
            Liu, Y., San Liang, X., & Weisberg, R. H. (2007),
            Rectification of the Bias in the Wavelet Power Spectrum,
            Journal of Atmospheric and Oceanic Technology.

        """
        opt = self.normalization_option
        cwt_coefs_n = deepcopy(cwt_coefs)
        if opt == 0:
            pass
        elif self.wavelet == "morl" and opt == 1:
            exp1 = (math.pi ** (-0.25)) / np.sqrt(2 * self.scales)
            exp2 = math.exp((1 / 4) * (self.w0 - math.sqrt(self.w0 ** 2 + 2)) ** 2)
            C = exp1 * exp2
            cwt_coefs_n[:, :, :] = np.abs(
                np.divide(cwt_coefs_n[:, :, :], C[np.newaxis, :, np.newaxis])
            )
        elif opt == 2:
            for i in range(cwt_coefs_n.shape[2]):
                for j in range(cwt_coefs_n.shape[1]):
                    s = self.scales[j]
                    cwt_coefs_n[:, j, i] = ((cwt_coefs_n[:, j, i]) ** 2) / s
        else:
            raise ValueError(
                f"For {self.wavelet}, unkown option {opt} is given for normalization."
            )
        return cwt_coefs_n
