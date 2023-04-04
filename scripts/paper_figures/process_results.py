import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker


class BehaviorData:
    def __init__(self, fps=30, behaviors=None):
        self.FPS = fps
        self.BEHAVIORS = behaviors if behaviors else []

    def get_time_stamp(self, idx, date="2022-01-01", shift_hours=10):
        sec_total = idx // self.FPS
        second = sec_total % 60
        minute = (sec_total // 60) % 60
        hour = sec_total // 3600
        stamp = str(int(hour)) + ":" + str(int(minute)) + ":" + str(int(second))
        return f"{date} {stamp}"

    def plot_raw_behavior(self, data, behavior, sd):
        g = sns.relplot(
            data=data,
            x='Idx',
            y=behavior,
            row='ExptNames',
            kind='line',
            palette='crest',
            height=2,
            aspect=10
        )

        for expt_name, ax in g.axes_dict.items():
            ax.text(.8, .85, expt_name, transform=ax.transAxes, fontweight='bold')
            ax.xaxis.grid(True)
        if sd == False:
            xticks = np.arange(start=0, stop=self.FPS * 60 * 60 * 16 + 1, step=self.FPS * 60 * 60 * 2)
        else:
            xticks = np.arange(start=0, stop=self.FPS * 60 * 60 * 6 + 1, step=self.FPS * 60 * 60 * 2)
        ax.set_xticks(xticks)
        zt_list = ['ZT' + str((tick + 10) % 24) for tick in range(0, len(xticks) * 2, 2)]
        ax.set_xticklabels(zt_list)
        ax.set_xlabel('ZT Time')
        g.set_titles("")
        g.tight_layout()
        g.fig.subplots_adjust(top=0.97)
        g.fig.suptitle('Behavior: ' + behavior)

    def save_fig(self, name, behavior, fig_path):
        fig_name = os.path.join(fig_path, name + 'behavior_' + behavior + '.pdf')
        svg_name = os.path.join(fig_path, name + 'behavior_' + behavior + '.svg')
        plt.savefig(fig_name, dpi=300)
        plt.savefig(svg_name)

    def plot_all_behaviors(self, data, name, fig_path):
        for behavior in self.BEHAVIORS:
            self.plot_raw_behavior(data, behavior, False)
            self.save_fig(name, behavior, fig_path)

    def pivot_and_plot(self, data, name, fig_path, rate):
        if rate[-1] == 'S':
            td = pd.Timedelta(rate)
            seconds = td.total_seconds()
        elif rate[-1] == 'T':
            td = pd.Timedelta(rate[-1] + 'm')
            seconds = td.total_seconds()

        for behavior in self.BEHAVIORS:
            df_pivoted = data.pivot(index='ExptNames', columns='TimeStamp', values=behavior)
            a4_dims = (25.7, 5.27)
            fig, ax = plt.subplots(figsize=a4_dims)
            plt.title(behavior)
            ax = sns.heatmap(df_pivoted, cmap='YlGnBu')

            locator = matplotlib.ticker.IndexLocator(base=(2 * 60 * 60) / seconds, offset=0)
            ax.xaxis.set_major_locator(locator)
            _, ZT_ticklabels = BehaviorData.generate_tick_data()
            ax.set_xticklabels(ZT_ticklabels)
            BehaviorData.save_fig(name, behavior, fig_path)

    def resample_df(data, rate):
        data_df_list = []
        for expt_name in data['ExptNames'].unique():
            sub_data = data[data['ExptNames'] == expt_name]
            data_ind_rs = sub_data[BEHAVIORS].resample(rate).mean()
            data_ind_rs['ExptNames'] = expt_name
            data_df_list.append(data_ind_rs)

            data_df_all_rs = pd.concat(data_df_list)

            return data_df_all_rs


    @staticmethod
    def generate_tick_data(FPS=30,sd=False):

        if sd == False:
            xticks = np.arange(start=0, stop=FPS * 60 * 60 * 16 + 1, step=FPS * 60 * 60 * 2)
        else:
            xticks = np.arange(start=0, stop=FPS * 60 * 60 * 6 + 1, step=FPS * 60 * 60 * 2)
        ZT_ticks = xticks
        ZT_ticklabels = ['ZT' + str((tick+10)%24) for tick in range(0,len(xticks)*2,2)]
        return ZT_ticks, ZT_ticklabels
