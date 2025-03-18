# Analyze an audio file, or all audio files in a directory.
# Output either Audacity label files or a CSV file depending on the --rtype argument.

import argparse
from datetime import datetime
import glob
import logging
import multiprocessing as mp
import os
from pathlib import Path
import random
import threading
import time
import yaml

import numpy as np
import torch

from core import audio
from core import cfg
from core import util
from model import main_model

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.reset()

    def reset(self):
        self.has_label = False
        self.scores = []     # predictions (one per segment)
        self.is_label = []   # True iff corresponding offset is a label

class Label:
    def __init__(self, class_name, score, start_time, end_time):
        self.class_name = class_name
        self.score = score
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path, overlap, device, num_threads=1, thread_num=1):
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.overlap = overlap
        self.num_threads = num_threads
        self.thread_num = thread_num
        self.device = device
        self.occurrences = {}

        # if no output path is specified, put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
            else:
                self.output_path = Path(self.input_path).parent
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    @staticmethod
    def _get_file_list(input_path):
        if os.path.isdir(input_path):
            return util.get_audio_files(input_path)
        elif util.is_audio_file(input_path):
            return [input_path]
        else:
            logging.error(f"Error: {input_path} is not a directory or an audio file")
            quit()

    # get class names and codes from the model, which gets them from the checkpoint
    def _get_class_infos(self):
        # we can use the class info from the trained model
        class_names = self.models[0].train_class_names
        class_codes = self.models[0].train_class_codes
        ignore_list = util.get_file_lines(cfg.misc.ignore_file)

        # create the ClassInfo objects
        class_infos = []
        for i, class_name in enumerate(class_names):
            class_infos.append(ClassInfo(class_name, class_codes[i], class_name in ignore_list))

        return class_infos

    # return the average prediction of all models in the ensemble
    def _call_models(self, specs):
        predictions = []
        for model in self.models:
            model.to(self.device)
            predictions.append(model.get_predictions(specs, self.device, use_softmax=False))

        # calculate and return the average across models
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    def _get_predictions(self, signal, rate):
        # if needed, pad the signal with zeros to get the last spectrogram
        total_seconds = signal.shape[0] / rate
        last_segment_len = total_seconds - cfg.audio.segment_len * (total_seconds // cfg.audio.segment_len)
        if last_segment_len > 0.5:
            # more than 1/2 a second at the end, so we'd better analyze it
            pad_amount = int(rate * (cfg.audio.segment_len - last_segment_len)) + 1
            signal = np.pad(signal, (0, pad_amount), 'constant', constant_values=(0, 0))

        max_end_seconds = max(0, (signal.shape[0] / rate) - cfg.audio.segment_len)
        end_seconds = max(max_end_seconds, 0)

        specs = self._get_specs(0, end_seconds)
        predictions = self._call_models(specs)
        for i in range(len(self.offsets)):
            for j in range(len(self.class_infos)):
                self.class_infos[j].scores.append(predictions[i][j])
                self.class_infos[j].is_label.append(False)
                if (self.class_infos[j].scores[-1] >= cfg.infer.min_score):
                    self.class_infos[j].has_label = True

    # get the list of spectrograms
    def _get_specs(self, start_seconds, end_seconds):
        increment = max(0.5, cfg.audio.segment_len - self.overlap)
        self.offsets = np.arange(start_seconds, end_seconds + 1.0, increment).tolist()
        self.raw_spectrograms = [0 for i in range(len(self.offsets))]
        specs = self.audio.get_spectrograms(self.offsets, segment_len=cfg.audio.segment_len, raw_spectrograms=self.raw_spectrograms)

        spec_array = np.zeros((len(specs), 1, cfg.audio.spec_height, cfg.audio.spec_width))
        for i in range(len(specs)):
            if specs[i] is not None:
                spec_array[i] = specs[i].reshape((1, cfg.audio.spec_height, cfg.audio.spec_width)).astype(np.float32)

        return spec_array

    def _analyze_file(self, file_path):
        logging.info(f"Thread {self.thread_num}: Analyzing {file_path}")

        # clear info from previous recording
        for class_info in self.class_infos:
            class_info.reset()

        signal, rate = self.audio.load(file_path)
        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # generate labels for one class at a time
        labels = []
        for class_info in self.class_infos:
            if class_info.ignore or not class_info.has_label:
                continue

            # set is_label[i] = True for any offset that qualifies in a first pass
            scores = class_info.scores
            for i in range(len(scores)):
                if scores[i] < cfg.infer.min_score or scores[i] == 0: # check for -p 0 case
                    continue

                class_info.is_label[i] = True

            # generate the labels
            for i in range(len(scores)):
                if class_info.is_label[i]:
                    end_time = self.offsets[i] + cfg.audio.segment_len
                    label = Label(class_info.code, scores[i], self.offsets[i], end_time)
                    labels.append(label)

        self._save_labels(labels, file_path)

    def _save_labels(self, labels, file_path):
        output_path = os.path.join(self.output_path, f'{Path(file_path).stem}_HawkEars.txt')
        logging.info(f"Thread {self.thread_num}: Writing {output_path}")
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.score:.3f}\n')
        except:
            logging.error(f"Unable to write file {output_path}")
            quit()

    # write a text file in YAML format, summarizing the inference and model parameters
    def _write_summary(self):
        time_struct = time.localtime(self.start_time)
        formatted_time = time.strftime("%H:%M:%S", time_struct)
        elapsed_time = util.format_elapsed_time(self.start_time, time.time())

        inference_key = "Inference / analysis"
        info = {inference_key: [
            {"version": util.get_version()},
            {"date": datetime.today().strftime('%Y-%m-%d')},
            {"start_time": formatted_time},
            {"elapsed": elapsed_time},
            {"device": self.device},
            {"num_threads": self.num_threads},
            {"min_score": cfg.infer.min_score},
            {"overlap": self.overlap},
            {"power": cfg.audio.power},
            {"segment_len": cfg.audio.segment_len},
            {"spec_height": cfg.audio.spec_height},
            {"spec_width": cfg.audio.spec_width},
            {"sampling_rate": cfg.audio.sampling_rate},
            {"win_length": cfg.audio.win_length},
            {"min_audio_freq": cfg.audio.min_audio_freq},
            {"max_audio_freq": cfg.audio.max_audio_freq},
        ]}

        # log info per model
        for i, model_path in enumerate(self.model_paths):
            model_info = [{"path": self.model_paths[i]}]
            model_info += self.models[i].summary()

            info[f"Model {i + 1}"] = model_info

        info_str = yaml.dump(info)
        info_str = "# Summary of HawkEars inference run in YAML format\n" + info_str
        with open(os.path.join(self.output_path, "HawkEars_summary.txt"), 'w') as out_file:
            out_file.write(info_str)

    def _get_models(self):
        self.model_paths = sorted(glob.glob(os.path.join(cfg.misc.main_ckpt_folder, "*.ckpt")))
        if len(self.model_paths) == 0:
            logging.error(f"Error: no checkpoints found in {cfg.misc.main_ckpt_folder}")
            quit()

        self.models = []
        for model_path in self.model_paths:
            model = main_model.MainModel.load_from_checkpoint(model_path, map_location=torch.device(self.device))
            model.eval() # set inference mode
            self.models.append(model)

    def run(self, file_list):
        self.start_time = time.time()
        torch.cuda.empty_cache()
        self._get_models()

        self.audio = audio.Audio(device=self.device)
        self.class_infos = self._get_class_infos()

        for file_path in file_list:
            self._analyze_file(file_path)

        # thread 1 writes a text file summarizing parameters etc.
        if self.thread_num == 1:
            self._write_summary()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='', help="Input path (single audio file or directory). No default.")
    parser.add_argument('-o', '--output', type=str, default='', help="Output directory to contain label files. Default is input path, if that is a directory.")
    parser.add_argument('--overlap', type=float, default=cfg.infer.spec_overlap_seconds, help=f"Seconds of overlap for adjacent 3-second spectrograms. Default = {cfg.infer.spec_overlap_seconds}.")
    parser.add_argument('-p', '--min_score', type=float, default=cfg.infer.min_score, help=f"Generate label if score >= this. Default = {cfg.infer.min_score}.")
    parser.add_argument('--threads', type=int, default=cfg.infer.num_threads, help=f'Number of threads. Default = {cfg.infer.num_threads}')
    parser.add_argument('--power', type=float, default=cfg.infer.audio_exponent, help=f'Power parameter to mel spectrograms. Default = {cfg.infer.audio_exponent}')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info("Initializing")

    num_threads = args.threads
    cfg.audio.power = args.power
    cfg.infer.min_score = args.min_score
    if cfg.infer.min_score < 0:
        logging.error("Error: min_score must be >= 0")
        quit()

    if torch.cuda.is_available():
        device = 'cuda'
        logging.info(f"Using CUDA")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logging.info(f"Using MPS")

    if cfg.infer.seed is not None:
        # reduce non-determinism
        torch.manual_seed(cfg.infer.seed)
        torch.cuda.manual_seed_all(cfg.infer.seed)
        random.seed(cfg.infer.seed)
        np.random.seed(cfg.infer.seed)

    file_list = Analyzer._get_file_list(args.input)
    if num_threads == 1:
        # keep it simple in case multithreading code has undesirable side-effects (e.g. disabling echo to terminal)
        analyzer = Analyzer(args.input, args.output, args.overlap, device, num_threads, 1)
        analyzer.run(file_list)
    else:
        # split input files into one group per thread
        file_lists = [[] for i in range(num_threads)]
        for i in range(len(file_list)):
            file_lists[i % num_threads].append(file_list[i])

        # for some reason using processes is faster than just using threads, but that disables output on Windows
        processes = []
        for i in range(num_threads):
            if len(file_lists[i]) > 0:
                analyzer = Analyzer(args.input, args.output, args.overlap, device, num_threads, i + 1)
                if os.name == "posix":
                    process = mp.Process(target=analyzer.run, args=(file_lists[i], ))
                else:
                    process = threading.Thread(target=analyzer.run, args=(file_lists[i], ))

                process.start()
                processes.append(process)

        # wait for processes to complete
        for process in processes:
            try:
                process.join()
            except Exception as e:
                logging.error(f"Caught exception: {e}")

    if os.name == "posix":
        os.system("stty echo")

    logging.info(f"Elapsed time = {util.format_elapsed_time(start_time, time.time())}")
