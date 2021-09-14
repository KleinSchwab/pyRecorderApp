import queue
import sys
import time
import warnings
from pathlib import Path
from threading import Thread, Timer

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:

    def __init__(self, device_id=None, block_len=0.05, max_memeory=-1, partition=False, playback=False):
        """

        Parameters
        ----------
        :param device_id:   int
                    Device id of the recording device.
                    Selecting the wrong device might still cause crashes.
        :param block_len:   int
                    Length of a recorded audi block in seconds.
                    Shorter audio block yield faster response (but require more CPU).
        :param max_memeory: int
                    Memory in byte before _manage_record is triggered to either discard or save audio data.
        :param partition: bool
                    Allows audio files to be stored as partition to e.g. use compressed file formats.
        :param playback: bool
                    Allows the playback of the recordings.
        """
        # get recording device
        self.device_id = device_id
        self.inputdevice = sd.query_devices(self.device_id, 'input')
        # try:
        #     self.inputdevice = sd.query_devices(self.device_id, 'input')
        # except Exception as e:
        #     self.inputdevice = sd.query_devices(self.device_id, 'input')
        #     print("Can't use device " + device_id + "Using device: " + self.inputdevice['hostapi'] + "instead.")
        self._samplerate = int(self.inputdevice['default_samplerate'])
        self._soundblock_q = queue.Queue(-1)
        self._channels = min(self.inputdevice['max_input_channels'], 1)
        self._block_length = block_len
        self._block_size = int(self._samplerate * block_len)  # calculate audio block size

        self._playback = playback  # flag to enable audio playback

        # setup memory management
        self.max_memory = max_memeory  # may memory stored in Q before saving to Disk
        self._mem_size = sys.getsizeof(self._soundblock_q)
        if max_memeory >= 0:
            self._memory_thread = None
            self._stime = .05

        self._stop_timer_thread = None

        self._partition = partition
        self._partition_ctr = 0

        self.filepath = None

        # internal variables
        self._input_stream = None
        self._rec = False
        self._default_fformat = ['wav', 'WAV']  # default file format

        self._rectime = 0.0
        self._prev_ab = None

    def _record_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block to store audio data from the input."""
        if status:
            print(status, file=sys.stderr)
        self._mem_size += sys.getsizeof(indata) + indata.nbytes
        self._prev_ab = indata.copy()
        self._soundblock_q.put(indata.copy())
        self._rectime += self._block_length

    def record_playback_callback(self, indata, outdata, frames, time, status):
        """
        This is called (from a separate thread) for each audio block to store
        and playback audio data from the input.
        """
        if status:
            print(status, file=sys.stderr)
        self._mem_size += sys.getsizeof(indata) + indata.nbytes
        self._prev_ab = indata.copy()
        self._soundblock_q.put(indata.copy())
        self._rectime += self._block_length
        outdata[:] = indata

    def _manage_record(self, now=False):
        """
        Function to manage memory saving RAM to disk when necessary or immediately when now is set to True.

        :param now: bool
                When True immediately saves the recorded data to disk.

        :return bool: Returns True if the Sound Block Q gets saved to disk and False otherwise.
        """
        if self._mem_size >= self.max_memory or now:
            audio_data = self._fuse_recording_q()
            self._save(audio_data)
            self._mem_size = sys.getsizeof(self._soundblock_q)
            return True
        else:
            return False

    def _recording_loop(self):
        """Event loop for managing memory (called in extra thread)"""
        while self._rec:
            self._manage_record()
            time.sleep(self._stime)

    def _fuse_recording_q(self):
        """
        Fuse all entries in the sound block queue into a single numpy array.

        :return np.ndarray: numpy array containing the audio amplitudes in the rows and channels in the columns
        """
        # check audioblock and allocate memory for numpy array
        if not self._soundblock_q.empty():
            nblocks = self._soundblock_q.qsize()
        else:
            return

        # preallocate memory for numpy array
        data_ptr = 0
        audio_np = np.zeros(((self._block_size + 1) * nblocks, self._channels), dtype=np.float32)

        for i in range(nblocks):
            try:
                audio_np[data_ptr:data_ptr + self._block_size, :] = self._soundblock_q.get(False)
                data_ptr += self._block_size + 1
            except Exception as e:
                print(e)
                continue
        self._soundblock_q.task_done()
        return audio_np

    def _save(self, audio_data: np.ndarray):
        """
        Saves the given audio disk. Depending on selected mode (partitioning) a new file will be created for each save
         or given the format allows it, the data will be added to a single file.

        :param audio_data:  np.ndarray
                    Audio data to be saved (rows are amplitudes, columns are channels).
        """
        if self.filepath is None:
            return
        if self.filepath.is_file() and not self._partition:
            with sf.SoundFile(str(self.filepath), mode='r+') as wfile:
                wfile.seek(0, sf.SEEK_END)
                wfile.write(audio_data)
        else:
            # check if file format is available
            suffix = self.filepath.suffix.split('.')[-1].casefold()
            avaiable_format = None
            for key in sf.available_formats().keys():
                if key.casefold() == suffix:
                    avaiable_format = key
                    break
            if avaiable_format is None:
                warnings.warn("Can't find available sound format for suffix {s}, using default {sd} instead."
                              .format(s=suffix, sd=self._default_fformat[0]))
                suffix = self._default_fformat[0]
                avaiable_format = self._default_fformat[1]
                self.filepath = self.filepath.with_suffix('.' + suffix)
                if self.filepath.is_file():
                    raise IOError('The soundfile {sf} already exists, please change the chosen sound format.'
                                  .format((str(self.filepath))))
            if self._partition:
                fp = str(self.filepath.parent) + self.filepath.stem + '({})'.format(self._partition_ctr) + \
                     self.filepath.suffix
                self._partition_ctr += 1
            else:
                fp = str(self.filepath)
            sf.write(fp, audio_data, self._samplerate, format=avaiable_format)  # writes to the new file

    def start_recording(self, filepath=None, max_time: float = -1.0):
        """
        Starts the recording.
        Internally start threads for audio Stream, audio recording, (data management, time management).

        :param filepath: str
                    Path to file if the audio data should be saved.
                    None if the data can be discarded.
        :param max_time: float
                    Maximum time for the recording in seconds (autostop afterwards).
                    Set to -1 to deactivate.
        """
        if self._rec:
            return

        if filepath is not None:
            self.filepath = Path(filepath)
        else:
            self.filepath = None

        self._rec = True
        self._rectime += 0.0
        if not self._playback:
            print(self.inputdevice)
            self._input_stream = sd.InputStream(device=self.device_id, channels=self._channels,
                                                blocksize=self._block_size,
                                                samplerate=self._samplerate, callback=self._record_callback)
        else:
            self._input_stream = sd.Stream(device=(self.device_id, None),
                                           channels=self._channels,
                                           blocksize=self._block_size,
                                           samplerate=self._samplerate,
                                           callback=self.record_playback_callback)
        if max_time > 0.0:
            self._stop_timer_thread = Timer(max_time, self.stop_recording)
            self._stop_timer_thread.setDaemon(True)

        if self.max_memory >= 0:
            self._memory_thread = Thread(target=self._recording_loop)
            self._memory_thread.setDaemon(True)

        self._input_stream.start()

        if self._memory_thread is not None:
            self._memory_thread.start()

        if self._stop_timer_thread is not None:
            self._stop_timer_thread.start()

    def stop_recording(self):
        """ Stops the recording and all running threads (and saves the data)."""
        if self._rec:
            self._rec = False
            if self._stop_timer_thread is not None:
                self._stop_timer_thread.cancel()
                self._stop_timer_thread = None
            self._input_stream.stop()
            if self.max_memory >= 0:
                self._memory_thread.join()
            self._manage_record(True)
            self._prev_ab = None
            self._rectime = 0.0

    def is_recording(self) -> bool:
        """ Returns a boolean if the recording is active. """
        return self._rec

    def get_sq_mem(self) -> int:
        """
         :return int
            Returns the estimated memory usage of the sound block queue in bytes.
         """
        return self._mem_size

    def get_rec_time(self):
        """
        :return float
            Returns the recording time in seconds (-1 if not recording)."""
        if self.is_recording():
            return self._rectime
        else:
            return -1

    def get_live_audio(self):
        """

        :return: np.ndarray
            Returns the last recorded audio block (from RAM).
        """
        if self._soundblock_q.qsize() > 0:
            return self._soundblock_q.queue[-1]
        else:
            return None

    def get_metadata(self):
        """

        :return: dict
            Dictionary containing metadata of the current recorder settings.
        """
        return {
            "samplerate": self._samplerate,
            "blocksize": self._block_size,
            "blocklength": self._block_length,
            "recording": self._rec,
            "recording_time": self._rectime,
            "memeory_usage": self._mem_size}
