import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from threading import Thread, Lock
from .constants import VIEW_BUFFER, VIEW_SUBSAMPLE, LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK


def view(window, scale, refresh, figure, backend, data_source="EEG"):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    print(f"Looking for a {data_source} stream...")
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError(f"Can't find {data_source} stream."))
    print(f"Found {len(streams)} {data_source} stream(s).")
    
    # Print detailed stream info
    for i, stream in enumerate(streams):
        print(f"Stream {i+1} details:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Channel count: {stream.channel_count()}")
        print(f"  Sampling rate: {stream.nominal_srate()}")
        print(f"  Source ID: {stream.source_id()}")
    
    print("Start acquiring data.")

    # Create the viewer with the stream
    lslv = LSLViewer(streams[0], window, scale, data_source=data_source)
    
    help_str = """
                toggle filter : d
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    
    # Start collecting data in background
    lslv.start()
    
    # Start the plotting (this will block until plot is closed)
    lslv.start_plotting()


class LSLViewer():
    def __init__(self, stream, window, scale, dejitter=True, data_source="EEG"):
        """Initialize the LSL Viewer"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.data_source = data_source
        self.filt = True
        self.started = False
        self.lock = Lock()  # For thread-safe data access
        
        # Set chunk size based on data source
        if data_source == "EEG":
            self.chunk_size = LSL_EEG_CHUNK
        elif data_source == "PPG":
            self.chunk_size = LSL_PPG_CHUNK
        elif data_source == "ACC":
            self.chunk_size = LSL_ACC_CHUNK
        elif data_source == "GYRO":
            self.chunk_size = LSL_GYRO_CHUNK
        else:
            self.chunk_size = LSL_EEG_CHUNK  # Default
            
        # Create inlet
        self.inlet = StreamInlet(stream, max_chunklen=self.chunk_size)
        
        # Get stream info
        info = self.inlet.info()
        description = info.desc()
        self.sfreq = info.nominal_srate()
        self.n_chan = info.channel_count()
        
        # Get channel names
        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]
        for i in range(self.n_chan - 1):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))
        self.ch_names = ch_names
        
        # Set up buffer for data collection
        buffer_size = int(self.sfreq * window * 2)  # Double window size for buffer
        self.buffer_size = buffer_size
        self.time_buffer = np.zeros(buffer_size)
        self.data_buffer = np.zeros((self.n_chan, buffer_size))
        self.buffer_idx = 0
        
        # Only initialize filters for EEG data
        if self.data_source == "EEG":
            self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                            pass_zero=False)
            self.af = [1.0]
            zi = lfilter_zi(self.bf, self.af)
            self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
        else:
            # For non-EEG data, we don't need filtering
            self.filt = False

    def update_buffer(self):
        """Update buffer with new data from the inlet"""
        try:
            while self.started:
                # Get data from inlet
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0, max_samples=self.chunk_size)
                
                if timestamps and len(timestamps) > 0:
                    # Convert to numpy arrays
                    timestamps = np.array(timestamps)
                    samples = np.array(samples)
                    
                    # Debug incoming data
                    if self.data_source == "PPG":
                        print(f"Received {len(timestamps)} PPG samples with shape: {samples.shape}")
                        print(f"Sample values: {samples[:5] if len(samples) >= 5 else samples}")
                    
                    # Process samples based on data type
                    if self.data_source in ["ACC", "GYRO"]:
                        # Ensure samples have the right shape
                        if samples.ndim == 1:
                            samples = samples.reshape(1, -1)
                        
                        # Just take the first 3 columns (X, Y, Z)
                        if samples.shape[1] >= 3:
                            samples = samples[:, :3]
                    
                    # Apply filtering for EEG data if needed
                    if self.data_source == "EEG" and self.filt and hasattr(self, 'bf'):
                        filt_samples, self.filt_state = lfilter(
                            self.bf, self.af, samples, axis=0, zi=self.filt_state)
                        samples = filt_samples
                    
                    # Add to buffer with thread safety
                    with self.lock:
                        for i, timestamp in enumerate(timestamps):
                            if i < len(samples):
                                # Calculate buffer position
                                idx = self.buffer_idx
                                
                                # Store timestamp
                                self.time_buffer[idx] = timestamp
                                
                                # Store sample data (transposed to match expected format)
                                if samples.ndim == 1:
                                    # Single sample case
                                    self.data_buffer[0, idx] = samples[i]
                                    if self.data_source == "PPG":
                                        print(f"Stored single PPG sample: {samples[i]}")
                                else:
                                    # Multiple channel case
                                    sample = samples[i]
                                    for ch in range(min(self.n_chan, len(sample))):
                                        self.data_buffer[ch, idx] = sample[ch]
                                    
                                    if self.data_source == "PPG":
                                        print(f"Stored multi-channel PPG sample: {sample[:self.n_chan]}")
                                
                                # Move buffer index
                                self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
                else:
                    # No data received, sleep a bit
                    if self.data_source == "PPG":
                        print("No PPG data received in this iteration")
                    sleep(0.1)
        except Exception as e:
            print(f"Error in update_buffer: {e}")

    def start_plotting(self):
        """
        Displays live plots for the chosen stream using matplotlib's FuncAnimation.
        """
        # Different plot setup based on data type
        if self.data_source == "EEG":
            # Special case for EEG - use single plot with stacked channels
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            lines = []
            
            # Create a line for each EEG channel
            for ii in range(self.n_chan):
                line, = ax.plot([], [], lw=1)
                lines.append(line)
            
            # Set up EEG plot
            ax.set_ylim(-self.n_chan + 0.5, 0.5)
            ticks = np.arange(0, -self.n_chan, -1)
            ax.set_yticks(ticks)
            
            # Initialize with empty labels
            ticks_labels = [f"{self.ch_names[ii]}" for ii in range(self.n_chan)]
            ax.set_yticklabels(ticks_labels)
            
            ax.set_xlabel('Time (s)')
            ax.set_xlim(-self.window, 0.1)
            ax.xaxis.grid(False)
            fig.suptitle(f"EEG Data")
            
            # Single axis for EEG
            axs = [ax]
            channels = self.n_chan
            
        else:
            # For ACC, GYRO, PPG use subplots for each channel
            if self.data_source in ["ACC", "GYRO"]:
                channels = 3  # X, Y, Z
            elif self.data_source == "PPG":
                channels = min(3, self.n_chan)  # Up to 3 PPG channels
            else:
                channels = self.n_chan
            
            fig, axs = plt.subplots(channels, 1, figsize=(10, 8), sharex=True)
            # Ensure axs is always a list/array for consistent indexing
            if channels == 1:
                axs = [axs]

            # Create line objects for blitting
            lines = []
            
            # Set fixed x-axis limits for time window
            time_window = self.window  # Show only the last N seconds
            
            # Set channel labels based on the selected stream
            if self.data_source == "ACC":
                labels = ["Acc X", "Acc Y", "Acc Z"]
            elif self.data_source == "GYRO":
                labels = ["Gyr X", "Gyr Y", "Gyr Z"]
            elif self.data_source == "PPG":
                labels = ["PPG IR", "PPG Red", "PPG Green"][:channels]
            else:
                labels = [f"Ch {i+1}" for i in range(channels)]
            
            # Initialize lines
            for i, lbl in enumerate(labels[:channels]):
                line, = axs[i].plot([], [], label=lbl)
                lines.append(line)
                axs[i].set_ylabel(lbl)
                axs[i].grid(True)
                
                # Set fixed y-axis limits based on data type
                if self.data_source == "ACC":
                    axs[i].set_ylim(-2, 2)
                elif self.data_source == "GYRO":
                    axs[i].set_ylim(-250, 250)
                elif self.data_source == "PPG":
                    axs[i].set_ylim(0, 4000)
            
            # Set fixed x-axis limits
            for ax in axs:
                ax.set_xlim(-time_window, 0.1)
            
            axs[-1].set_xlabel('Time (s)')
            fig.suptitle(f"{self.data_source} (last {time_window}s)")
        
        # Connect key press events
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        def update(_):
            with self.lock:
                idx = self.buffer_idx
                # Use np.roll for more efficient circular buffer handling
                t = np.roll(self.time_buffer, -idx)
                d = np.roll(self.data_buffer, -idx, axis=1)
            
            # Debug data buffer state
            if self.data_source == "PPG":
                non_zero_count = np.count_nonzero(d)
                if non_zero_count > 0:
                    print(f"PPG data buffer has {non_zero_count} non-zero values")
                    print(f"Sample values: {d[:, -10:] if d.shape[1] >= 10 else d}")
                else:
                    print("PPG data buffer is empty (all zeros)")
            
            # Filter data for the time window
            if len(t) > 0 and np.any(t != 0):  # Check if we have valid timestamps
                last_timestamp = t[-1]
                mask = t >= (last_timestamp - self.window)
                # Ensure we have at least some data points even if time window is not filled yet
                if np.sum(mask) > 0:
                    t_plot = t[mask]
                    d_plot = d[:, mask]
                    if self.data_source == "PPG":
                        print(f"Plotting {len(t_plot)} PPG data points")
                else:
                    # If no data in time window yet, use all available non-zero data
                    non_zero_mask = t != 0
                    t_plot = t[non_zero_mask] if np.any(non_zero_mask) else np.array([0])
                    d_plot = d[:, non_zero_mask] if np.any(non_zero_mask) else np.zeros((channels, 1))
                    if self.data_source == "PPG" and np.any(non_zero_mask):
                        print(f"Using {np.sum(non_zero_mask)} non-zero PPG data points")
            else:
                t_plot = np.array([0])
                d_plot = np.zeros((channels, 1))
                if self.data_source == "PPG":
                    print("No valid timestamps found for PPG data")

            # Normalize time to show seconds from current time
            if len(t_plot) > 0 and t_plot[-1] != 0:
                t_normalized = t_plot - t_plot[-1]
            else:
                t_normalized = np.array([0])
            
            # Update line data based on data source
            if self.data_source == "EEG":
                # Special handling for EEG - stacked channels
                for ii in range(self.n_chan):
                    if ii < d_plot.shape[0]:
                        lines[ii].set_data(t_normalized, d_plot[ii] / self.scale - ii)
                
                # Update impedance values in y-axis labels
                if len(d_plot) > 0 and d_plot.shape[1] > 0:
                    impedances = np.std(d_plot, axis=1)
                    ticks_labels = ['%s - %.2f' % (self.ch_names[ii], impedances[ii])
                                    for ii in range(self.n_chan)]
                    axs[0].set_yticklabels(ticks_labels)
            else:
                # Normal handling for other data types
                for i, line in enumerate(lines):
                    if i < d_plot.shape[0]:  # Ensure we don't exceed data dimensions
                        line.set_data(t_normalized, d_plot[i])
            
            return lines

        # Set up the animation with blitting for better performance
        self.anim = animation.FuncAnimation(
            fig,
            update,
            interval=100,
            blit=True,
            save_count=100,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()
    
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == '/':
            self.scale *= 1.2
            print(f"Scale increased to {self.scale}")
        elif event.key == '*':
            self.scale /= 1.2
            print(f"Scale decreased to {self.scale}")
        elif event.key == '+':
            self.window += 1
            print(f"Window increased to {self.window}s")
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
                print(f"Window decreased to {self.window}s")
        elif event.key == 'd':
            # Only toggle filter for EEG data
            if self.data_source == "EEG":
                self.filt = not(self.filt)
                print(f"Filtering {'enabled' if self.filt else 'disabled'}")

    def start(self):
        """Start the background data collection thread"""
        self.started = True
        self.thread = Thread(target=self.update_buffer)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, close_event=None):
        """Stop the background thread"""
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
