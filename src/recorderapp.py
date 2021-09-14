import time
import datetime
import math

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog
from audio.utils import BlitManager
from audio.audiomodel import AudioRecorder


class RecorderApp:
    def __init__(self, t_freq=.03, device_id=0, block_len=.05):
        """


        Parameters
        ----------
        :param t_freq:      float
                            is measured in seconds, determines the udpate frequency of the main loop
        :param device_id:   int
                            device id of the recording device
                            selecting the wrong device might still cause crashes
        :param block_len:   float
                            length of a recorded audi block in seconds
                            shorter audio block yield faster response (but require more CPU)
        """
        self.t_freq = int(t_freq*1000)
        self.dt_last_update = 0
        self.ap_data_sr = 1  # internal variable for scaling
        self.ap_plotting = True  # enable/disable plotting
        # create AudiRecorder Model
        #ToDo: setting for audio recorder
        self.audio_recorder = AudioRecorder(device_id=device_id, playback=False, max_memeory=1000000, partition=False,
                                            block_len=block_len)

        self.window = tk.Tk()

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.ui_scalefactor = (1.2 * self.screen_width) / 1920
        self.font_standard = (None, round(self.ui_scalefactor * 12))
        self.font_big = (None, round(self.ui_scalefactor * 16))
        self.font_large = (None, round(self.ui_scalefactor * 20))
        self.font_titel = (None, round(self.ui_scalefactor * 24), 'bold')

        self.window.pack_propagate(0)
        self.window.resizable(0, 0)
        self.window.geometry(str(self.screen_width) + "x" + str(self.screen_height))

        self.fs = True
        self.window.attributes("-fullscreen", self.fs)
        time.sleep(0)  # delay to _update size of main window
        self.window.bind("<F11>", self._toggle_fscreen)
        self.window.bind("<Escape>", self._quit_fscreen)

        # build the main window and the plot
        self._build_view()

        # setup other variables
        self.save_folder = None

        self.window.after(self.t_freq, self._update)
        self.window.mainloop()

        # free widgets and windows bc tkinter doesn't like being called in spared threads
        self.menu_buttons = None
        self.window = None
        self.main_frame = None
        self.control_frame = None
        self.plot_frame = None
        self.control_frame_grid = None


    def _build_view(self):
        """ Builds the main view and sets up the GUI elements. """
        self.main_frame = tk.Frame(self.window)

        self.main_frame.pack(side=tk.LEFT)
        self.control_frame = tk.Frame()
        self.plot_frame = tk.Frame()

        height = self.screen_height
        width = self.screen_width

        self.control_frame = tk.Frame(self.main_frame, height=height, width=int(0.1 * width))
        self.control_frame.pack(side=tk.LEFT)
        self.control_frame_grid = []
        gird_rows = 6
        for row in range(gird_rows):
            f = tk.Frame(self.control_frame, height=int((height * 0.95) / gird_rows),
                         width=int((0.1 * width) * 0.95))
            f.pack_propagate(0)  # don't shrink the frame
            f.grid(row=row, column=0, padx=int(int(0.05 * width) * 0.05),
                   pady=int((height * 0.05) / (2 * gird_rows)))
            self.control_frame_grid.append(f)

        self.save_record_button = tk.Button(self.control_frame_grid[0],
                                            text='Save Location', font=self.font_standard, command=self._select_saveloc)
        self.save_record_button.pack(fill=tk.BOTH, expand=1)

        self.start_rec_btn = tk.Button(self.control_frame_grid[1], text='Start Recording', font=self.font_standard,
                                       command=self._start_rec)
        self.start_rec_btn.pack(fill=tk.BOTH, expand=1)

        self.stop_rec_btn = tk.Button(self.control_frame_grid[2], text='Stop Recording', font=self.font_standard,
                                      command=self._stop_rec)
        self.stop_rec_btn.pack(fill=tk.BOTH, expand=1)

        self.recroding_label = tk.Label(self.control_frame_grid[3], text="Not Recording", font=self.font_big)
        self.recroding_label.pack(side='top')

        self.rec_time_text = tk.StringVar()
        self.rec_time_label = tk.Label(self.control_frame_grid[3], textvariable=self.rec_time_text,
                                       font=self.font_big, bg='white', width=width)
        self.rec_time_label.pack(side='top')
        self._set_record_time()

        self.plot_btn = tk.Button(self.control_frame_grid[4], text='Plot On', relief='raised', font=self.font_standard,
                                  command=self._toggle_plot)
        self.plot_btn.pack(fill=tk.BOTH, expand=1)

        self.exit_button = tk.Button(self.control_frame_grid[5], text='Exit', font=self.font_standard,
                                     command=self._quit)
        self.exit_button.pack(fill=tk.BOTH, expand=1)

        self.plot_frame = tk.Frame(self.main_frame, height=height, width=width - int(0.1 * width))
        self.plot_frame.pack_propagate(0)
        self.plot_frame.pack(side=tk.RIGHT)

        # build audio plot
        self.ap_fig = plt.Figure(dpi=72, tight_layout=True)
        self.fig_canvas = FigureCanvasTkAgg(self.ap_fig, master=self.plot_frame)
        self.ap_ax = self.ap_fig.add_subplot(111)
        mdata = self.audio_recorder.get_metadata()
        # calculate the downsampling of data for faster display frequency
        self.ap_data_sr = max(1, round(self.ap_data_sr * (width - self.control_frame.winfo_width()) /
                                  math.floor(mdata["blocksize"])))

        x = np.linspace(0, mdata["blocklength"], mdata["blocksize"])[::self.ap_data_sr]
        (self.ap_ln,) = self.ap_ax.plot(x, np.zeros_like(x), animated=True, linewidth=2.2*self.ui_scalefactor)
        self.ap_ax.set_xlim([0.0, mdata["blocklength"]])
        self.ap_ax.set_ylim([-1.0, 1.0])
        self.ap_ax.grid(linestyle='--', linewidth=0.5)
        self.ap_ax.set_title(label="Last Recorded Audio Block", fontsize=28*self.ui_scalefactor, fontweight='bold')
        self.ap_ax.set_xlabel("Time $\it{t}$ in seconds", fontsize=22*self.ui_scalefactor )
        self.ap_ax.set_ylabel("Amplitude", fontsize=22*self.ui_scalefactor)
        # set font size
        self.ap_ax.tick_params(axis='both', which='major', labelsize=16*self.ui_scalefactor)
        self.ap_ax.tick_params(axis='both', which='minor', labelsize=14*self.ui_scalefactor)

        self.ap_bm = BlitManager(self.ap_fig.canvas, [self.ap_ln])
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.fig_canvas.draw()
        # store background for redraw

    def _update_audio_plot(self):
        """ Updates the audio Plot with the sound data of the current recording."""
        data = self.audio_recorder.get_live_audio()
        if data is not None:
            self.ap_ln.set_ydata(data[::self.ap_data_sr])
            # tell the blitting manager to do its thing
            self.ap_bm.update()

    def _set_record_time(self, t=None):
        """
        Set the time on the rec_time_label.

        Parameters
        ----------
        :param t: float
                  time in seconds
        """
        if t is None:
            self.rec_time_text.set("-:--:--:--")
        elif t < 0:
            self.rec_time_text.set("-:--:--:--")
        else:
            self.rec_time_text.set(str(datetime.timedelta(milliseconds=t*1000))[:-4])

    def _toggle_fscreen(self, event):
        """ Toggle function to enable/disable fullscreen mode."""
        self.fs = not self.fs
        self.window.attributes("-fullscreen", self.fs)

    def _quit_fscreen(self, event):
        """ Function to quit fullscreen."""
        self.fs = False
        self.window.attributes("-fullscreen", self.fs)

    def _update(self):
        """Update function or main loop of the App."""
        t = time.time()
        if self.audio_recorder.is_recording():
            if self.ap_plotting:
                self._update_audio_plot()
            self._set_record_time(self.audio_recorder.get_rec_time())
        self.dt_last_update = round((time.time()-t) * 1000)
        adjusted_t_freq = max(1, self.t_freq-self.dt_last_update)
        self.window.after(adjusted_t_freq, self._update)

    def _start_rec(self):
        """(Button)Function to start the recording."""
        if self.save_folder is not None:
            # filepath is coded with current time and date
            filepath = self.save_folder / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav")
            self.audio_recorder.start_recording(str(filepath))
        else:
            self.audio_recorder.start_recording()
        self.recroding_label.config(text='Recording', fg='red')
        self._set_record_time(self.audio_recorder.get_rec_time())

    def _stop_rec(self):
        """(Button)Function to stop the recording."""
        self.audio_recorder.stop_recording()
        self.recroding_label.config(text='Not Recording', fg='black')
        self._set_record_time()

    def _select_saveloc(self):
        """(Button)Function to select the Folder for saving audio Files."""
        path = filedialog.askdirectory()
        if path is not '':
            self.save_folder = Path(path)
        else:
            self.save_folder = None

    def _toggle_plot(self):
        """Toggle (Button)Function to switch the live poltting On/Off (reduce computation)."""
        if self.plot_btn.config('relief')[-1] == 'sunken':
            self.plot_btn.config(text="Plot On", relief="raised")
            self.ap_plotting = True
        else:
            self.plot_btn.config(text="Plot Off", relief="sunken")
            self.ap_plotting = False

    def _quit(self):
        """Function to stop the Application."""
        self._stop_rec()
        self.window.destroy()


def main():
    recorder_app = RecorderApp(30)


if __name__ == "__main__":
    main()
