import numpy as np
import matplotlib
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from threading import Thread
from .constants import VIEW_BUFFER, VIEW_SUBSAMPLE, LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK


def view(window, scale, refresh, figure, backend, data_source="EEG"):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.int16(figure.split('x'))

    print(f"Looking for a {data_source} stream...")
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError(f"Can't find {data_source} stream."))
    print("Start acquiring data.")

    # Create appropriate number of subplots based on data source
    if data_source == "EEG":
        fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
        fig.suptitle('EEG Data', fontsize=16)
    elif data_source == "PPG":
        fig, axes = matplotlib.pyplot.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('PPG Data', fontsize=16)
    elif data_source in ["ACC", "GYRO"]:
        fig, axes = matplotlib.pyplot.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle(f'{data_source} Data', fontsize=16)
    
    lslv = LSLViewer(streams[0], fig, axes, window, scale, data_source=data_source)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toogle full screen : f
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    lslv.start()
    matplotlib.pyplot.show()


class LSLViewer():
    def __init__(self, stream, fig, axes, window, scale, dejitter=True, data_source="EEG"):
        """Init"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.data_source = data_source
        
        # Set chunk size based on data source
        if data_source == "EEG":
            chunk_size = LSL_EEG_CHUNK
        elif data_source == "PPG":
            chunk_size = LSL_PPG_CHUNK
        elif data_source == "ACC":
            chunk_size = LSL_ACC_CHUNK
        elif data_source == "GYRO":
            chunk_size = LSL_GYRO_CHUNK
        else:
            chunk_size = LSL_EEG_CHUNK  # Default
            
        self.inlet = StreamInlet(stream, max_chunklen=chunk_size)
        self.filt = True
        self.subsample = VIEW_SUBSAMPLE

        info = self.inlet.info()
        description = info.desc()

        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        for i in range(self.n_chan - 1):  # Subtract 1 because we already got the first channel
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        self.ch_names = ch_names

        fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.fig = fig
        self.axes = axes

        sns.despine(left=True)

        # Initialize data arrays based on data source
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)
        
        if self.data_source == "EEG":
            self.data = np.zeros((self.n_samples, self.n_chan))
            impedances = np.std(self.data, axis=0)
            lines = []
            
            for ii in range(self.n_chan):
                line, = self.axes.plot(self.times[::self.subsample],
                                  self.data[::self.subsample, ii] - ii, lw=1)
                lines.append(line)
            self.lines = lines
            
            self.axes.set_ylim(-self.n_chan + 0.5, 0.5)
            ticks = np.arange(0, -self.n_chan, -1)
            
            self.axes.set_xlabel('Time (s)')
            self.axes.xaxis.grid(False)
            self.axes.set_yticks(ticks)
            
            ticks_labels = ['%s - %.1f' % (self.ch_names[ii], impedances[ii])
                            for ii in range(self.n_chan)]
            self.axes.set_yticklabels(ticks_labels)
            
        elif self.data_source == "PPG":
            # For PPG, we have 3 channels
            self.data = np.zeros((self.n_samples, self.n_chan))
            lines = []
            
            # Create a separate subplot for each PPG channel
            if isinstance(self.axes, np.ndarray):
                for ii in range(min(3, self.n_chan)):
                    line, = self.axes[ii].plot(self.times[::self.subsample],
                                          self.data[::self.subsample, ii], lw=1)
                    lines.append(line)
                    self.axes[ii].set_ylabel(f'{self.ch_names[ii]} (a.u.)')
                    self.axes[ii].set_xlabel('Time (s)' if ii == 2 else '')
                    self.axes[ii].set_ylim(0, self.scale * 10)  # PPG values are typically higher
            else:
                # Fallback to single plot if axes is not an array
                for ii in range(min(3, self.n_chan)):
                    line, = self.axes.plot(self.times[::self.subsample],
                                      self.data[::self.subsample, ii], lw=1, 
                                      label=f'{self.ch_names[ii]}')
                    lines.append(line)
                self.axes.set_ylabel('Amplitude (a.u.)')
                self.axes.set_xlabel('Time (s)')
                self.axes.legend()
                
            self.lines = lines
            
        elif self.data_source in ["ACC", "GYRO"]:
            # For ACC and GYRO, we have 3 axes (X, Y, Z)
            self.data = np.zeros((self.n_samples, 3))  # X, Y, Z
            lines = []
            
            # Labels for axes
            axis_labels = ['X', 'Y', 'Z']
            y_unit = 'g' if self.data_source == "ACC" else 'deg/s'
            
            # Create a separate subplot for each axis
            if isinstance(self.axes, np.ndarray):
                for ii in range(3):
                    line, = self.axes[ii].plot(self.times[::self.subsample],
                                          self.data[::self.subsample, ii], lw=1)
                    lines.append(line)
                    self.axes[ii].set_ylabel(f'{axis_labels[ii]} ({y_unit})')
                    self.axes[ii].set_xlabel('Time (s)' if ii == 2 else '')
                    self.axes[ii].set_ylim(-self.scale / 10, self.scale / 10)  # ACC/GYRO values are typically smaller
            else:
                # Fallback to single plot if axes is not an array
                for ii in range(3):
                    line, = self.axes.plot(self.times[::self.subsample],
                                      self.data[::self.subsample, ii], lw=1, 
                                      label=f'{axis_labels[ii]}')
                    lines.append(line)
                self.axes.set_ylabel(f'Amplitude ({y_unit})')
                self.axes.set_xlabel('Time (s)')
                self.axes.legend()
                
            self.lines = lines

        self.display_every = int(0.2 / (12 / self.sfreq))

        # Only initialize filters for EEG data
        if self.data_source == "EEG":
            self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                            pass_zero=False)
            self.af = [1.0]

            zi = lfilter_zi(self.bf, self.af)
            self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
            self.data_f = np.zeros((self.n_samples, self.n_chan))
        else:
            # For non-EEG data, we don't need filtering
            self.filt = False

    def update_plot(self):
        k = 0
        try:
            while self.started:
                # Get chunk size based on data source
                if self.data_source == "EEG":
                    chunk_size = LSL_EEG_CHUNK
                elif self.data_source == "PPG":
                    chunk_size = LSL_PPG_CHUNK
                elif self.data_source == "ACC":
                    chunk_size = LSL_ACC_CHUNK
                elif self.data_source == "GYRO":
                    chunk_size = LSL_GYRO_CHUNK
                else:
                    chunk_size = LSL_EEG_CHUNK  # Default
                
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                           max_samples=chunk_size)

                if timestamps and len(timestamps) > 0:
                    # Ensure timestamps is a 1D array
                    timestamps = np.array(timestamps).flatten()
                    
                    if self.dejitter:
                        timestamps = np.float64(np.arange(len(timestamps)))
                        timestamps /= self.sfreq
                        timestamps += self.times[-1] + 1. / self.sfreq
                    
                    # Ensure both arrays have the same dimensions before concatenation
                    self.times = np.concatenate([self.times, timestamps])
                    self.n_samples = int(self.sfreq * self.window)
                    self.times = self.times[-self.n_samples:]
                    
                    # Process data based on data source
                    if self.data_source in ["ACC", "GYRO"]:
                        # Convert samples to numpy array if it's not already
                        samples = np.array(samples)
                        
                        # For ACC and GYRO, data comes as [sample1, sample2, ...] where each sample is [x, y, z]
                        # We need to reshape to get separate x, y, z channels
                        if len(samples) > 0 and samples.shape[1] == 3:  # Make sure we have 3 dimensions (x, y, z)
                            # Reshape samples to match our data structure
                            reshaped_samples = np.zeros((len(samples), 3))
                            for i in range(len(samples)):
                                reshaped_samples[i, 0] = samples[i, 0]  # X
                                reshaped_samples[i, 1] = samples[i, 1]  # Y
                                reshaped_samples[i, 2] = samples[i, 2]  # Z
                            
                            self.data = np.vstack([self.data, reshaped_samples])
                            self.data = self.data[-self.n_samples:]
                            
                            # No filtering for ACC/GYRO data
                            plot_data = self.data
                            
                            k += 1
                            if k == self.display_every:
                                # Update each axis plot
                                if isinstance(self.axes, np.ndarray):
                                    for ii in range(3):
                                        # Get current data for this axis
                                        current_data = plot_data[::self.subsample, ii]
                                        
                                        # Update line data
                                        self.lines[ii].set_xdata(self.times[::self.subsample] - self.times[-1])
                                        self.lines[ii].set_ydata(current_data)
                                        
                                        # Dynamically adjust y-limits based on current data
                                        if len(current_data) > 0:
                                            data_min, data_max = np.min(current_data), np.max(current_data)
                                            data_range = max(data_max - data_min, 0.001)  # Avoid zero range
                                            
                                            # Center around the mean with padding
                                            avg_value = np.mean(current_data)
                                            y_min = avg_value - data_range * 1.2
                                            y_max = avg_value + data_range * 1.2
                                            
                                            # Apply the new limits
                                            self.axes[ii].set_ylim(y_min, y_max)
                                        
                                        self.axes[ii].set_xlim(-self.window, 0)
                                else:
                                    # Single plot case
                                    for ii in range(3):
                                        self.lines[ii].set_xdata(self.times[::self.subsample] - self.times[-1])
                                        self.lines[ii].set_ydata(plot_data[::self.subsample, ii])
                                    
                                    # Dynamically adjust y-limits for the combined plot
                                    all_data = plot_data[::self.subsample, :].flatten()
                                    if len(all_data) > 0:
                                        data_min, data_max = np.min(all_data), np.max(all_data)
                                        data_range = max(data_max - data_min, 0.001)
                                        avg_value = np.mean(all_data)
                                        y_min = avg_value - data_range * 1.2
                                        y_max = avg_value + data_range * 1.2
                                        self.axes.set_ylim(y_min, y_max)
                                    
                                    self.axes.set_xlim(-self.window, 0)
                                
                                self.fig.canvas.draw()
                                k = 0
                    
                    elif self.data_source == "PPG":
                        # Convert samples to numpy array if it's not already
                        samples = np.array(samples)
                        
                        # For PPG, we have regular samples
                        if len(samples) > 0:
                            self.data = np.vstack([self.data, samples])
                            self.data = self.data[-self.n_samples:]
                        
                        # No filtering for PPG data
                        plot_data = self.data
                        
                        k += 1
                        if k == self.display_every:
                            # Update each PPG channel plot
                            if isinstance(self.axes, np.ndarray):
                                for ii in range(min(3, self.n_chan)):
                                    # Get current data for this channel
                                    current_data = plot_data[::self.subsample, ii]
                                    
                                    # Update line data
                                    self.lines[ii].set_xdata(self.times[::self.subsample] - self.times[-1])
                                    self.lines[ii].set_ydata(current_data)
                                    
                                    # Dynamically adjust y-limits based on current data
                                    if len(current_data) > 0:
                                        data_min, data_max = np.min(current_data), np.max(current_data)
                                        data_range = max(data_max - data_min, 1.0)  # Avoid zero range
                                        
                                        # For PPG, we typically want to start from zero or slightly below
                                        y_min = max(0, data_min - data_range * 0.1)
                                        y_max = data_max + data_range * 0.1
                                        
                                        # Apply the new limits
                                        self.axes[ii].set_ylim(y_min, y_max)
                                    
                                    self.axes[ii].set_xlim(-self.window, 0)
                            else:
                                # Single plot case
                                for ii in range(min(3, self.n_chan)):
                                    self.lines[ii].set_xdata(self.times[::self.subsample] - self.times[-1])
                                    self.lines[ii].set_ydata(plot_data[::self.subsample, ii])
                                
                                # Dynamically adjust y-limits for the combined plot
                                all_data = plot_data[::self.subsample, :min(3, self.n_chan)].flatten()
                                if len(all_data) > 0:
                                    data_min, data_max = np.min(all_data), np.max(all_data)
                                    data_range = max(data_max - data_min, 1.0)
                                    y_min = max(0, data_min - data_range * 0.1)
                                    y_max = data_max + data_range * 0.1
                                    self.axes.set_ylim(y_min, y_max)
                                
                                self.axes.set_xlim(-self.window, 0)
                            
                            self.fig.canvas.draw()
                            k = 0
                    
                    else:  # EEG
                        # Convert samples to numpy array if it's not already
                        samples = np.array(samples)
                        
                        if len(samples) > 0:
                            self.data = np.vstack([self.data, samples])
                            self.data = self.data[-self.n_samples:]
                        
                        # Only apply filtering for EEG data
                        if hasattr(self, 'bf') and hasattr(self, 'af') and hasattr(self, 'filt_state'):
                            filt_samples, self.filt_state = lfilter(
                                self.bf, self.af,
                                samples,
                                axis=0, zi=self.filt_state)
                            self.data_f = np.vstack([self.data_f, filt_samples])
                            self.data_f = self.data_f[-self.n_samples:]
                        k += 1
                        if k == self.display_every:
                            if self.filt:
                                plot_data = self.data_f
                            elif not self.filt:
                                plot_data = self.data - self.data.mean(axis=0)
                            
                            for ii in range(self.n_chan):
                                self.lines[ii].set_xdata(self.times[::self.subsample] -
                                                        self.times[-1])
                                self.lines[ii].set_ydata(plot_data[::self.subsample, ii] /
                                                        self.scale - ii)
                                
                            impedances = np.std(plot_data, axis=0)
                            ticks_labels = ['%s - %.2f' % (self.ch_names[ii],
                                                        impedances[ii])
                                            for ii in range(self.n_chan)]
                            self.axes.set_yticklabels(ticks_labels)
                            self.axes.set_xlim(-self.window, 0)
                            self.fig.canvas.draw()
                            k = 0
                else:
                    sleep(0.2)
        except RuntimeError as e:
            raise

    def onclick(self, event):
        print((event.button, event.x, event.y, event.xdata, event.ydata))

    def OnKeypress(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.window += 1
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
        elif event.key == 'd':
            # Only toggle filter for EEG data
            if self.data_source == "EEG":
                self.filt = not(self.filt)
                print(f"Filtering {'enabled' if self.filt else 'disabled'}")

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update_plot)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, close_event):
        self.started = False
