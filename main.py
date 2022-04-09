import concurrent.futures
import functools
import glob
import multiprocessing
import itertools
import random
import threading
import time

import librosa
import numpy as np
import simpleaudio

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen


DEBUG = False
SEED = 0 if DEBUG else time.time()


# Must be stereo FLAC
class Audio:
    def __init__(self, samples, sample_rate, bit_depth):
        self.samples = samples
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth

    def __str__(self):
        return f"{self.sample_rate} Hz {self.bit_depth}-bit"


class Test:
    def __init__(self, file, sample_a, sample_b):
        self.file = file
        self.sample_a = sample_a
        self.sample_b = sample_b
        self.result = None

    def __str__(self):
        return f"A: {self.sample_a}, B: {self.sample_b}, Result: {self.result} ({self.file})"


class IntroScreen(Screen):
    pass


class TestScreen(Screen):
    pass


class ResultsScreen(Screen):
    pass


def truncate(y, sr, duration, offset):
    start = int(sr * offset)
    end = int(start + (sr * duration))
    return y[:, start:end]


def random_snippet(y, sr, duration=4):
    avail_seconds = max(0, (y.shape[1] / sr) - duration)
    offset = random.random() * avail_seconds
    return truncate(y, sr, duration, offset)


def change_bit_depth(y, bit_depth):
    max_val = (2 ** (bit_depth - 1)) - 1
    y *= max_val / np.max(np.abs(y))
    y = y.astype(np.int32)
    bytes = []
    bytes_per_sample = bit_depth // 8
    for i, b in zip(itertools.cycle(range(4)), y.tobytes()):
        if i < bytes_per_sample:
            bytes.append(b)
    return bytearray(bytes)


def resample32(y, orig_sr, target_sr, bit_depth):
    y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    y = np.column_stack((y[0], y[1]))
    audio = Audio(change_bit_depth(y, bit_depth), target_sr, bit_depth)
    return audio


class SnakeOilApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.screen_manager = None
        self.sample_rates = [22050, 32000, 44100, 48000, 96000]
        self.bit_depths = [16, 24]
        self.files = []
        self.audio = {}

        self.num_snippets = 3
        self.num_tests = 20
        self.cur_test = 0
        self.tests = []

    def prepare_audio(self):
        files = glob.glob("./audio/*.flac")
        if not files:
            raise ValueError("No audio files found in the audio subdirectory.")

        start_btn = self.screen_manager.get_screen("intro").ids.start_btn

        def bad_file(file, orig_sr, dt):
            if orig_sr < self.sample_rates[-1]:
                text = f'File "{file}"\nhas a sample rate {orig_sr} < {self.sample_rates[-1]} Hz and will be ignored.'
            else:
                text = f'File "{file}"\nis not stereo and will be ignored.'
            content = Button(text=text)
            popup = Popup(title="Bad File", content=content)
            content.bind(on_press=popup.dismiss)
            popup.open()

        ys, orig_srs = [], []
        for file in files:
            start_btn.text = f"Loading {file}..."
            y, orig_sr = librosa.load(file, sr=None, mono=False)
            if orig_sr < self.sample_rates[-1] or len(y.shape) != 2 or y.shape[0] != 2:
                Clock.schedule_once(functools.partial(bad_file, file, orig_sr))
                continue
            self.files.append(file)
            self.audio[file] = [[] for _ in range(self.num_snippets)]
            ys.append(y)
            orig_srs.append(orig_sr)

        load_pb = self.screen_manager.get_screen("intro").ids.load_pb
        num_audio_samples = (
            len(self.sample_rates)
            * len(self.bit_depths)
            * len(self.files)
            * self.num_snippets
        )
        increment = load_pb.max / (num_audio_samples + 1)

        def update_progress(text, dt):
            start_btn.text = text
            load_pb.value += increment

        def task(file, snippet_id, snippet, orig_sr, sr, bit_depth):
            text = f'Preparing snippet {snippet_id+1} from "{file}" @ {sr} Hz, {bit_depth}-bit...'
            Clock.schedule_once(functools.partial(update_progress, text))
            audio = resample32(snippet, orig_sr, sr, bit_depth)
            self.audio[file][snippet_id].append(audio)

        start = time.time()
        cpu_count = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as exe:
            for file, y, orig_sr in zip(self.files, ys, orig_srs):
                for snippet_id in range(self.num_snippets):
                    snippet = random_snippet(y, orig_sr)
                    for sr in self.sample_rates:
                        for bit_depth in self.bit_depths:
                            exe.submit(
                                task, file, snippet_id, snippet, orig_sr, sr, bit_depth
                            )
        if DEBUG:
            print("audio prep time:", time.time() - start)

    def prepare_tests(self):
        self.tests.clear()
        for test in range(self.num_tests):
            file = random.choice(self.files)
            snippet_list = random.choice(self.audio[file])
            sample_a = random.choice(snippet_list)
            # Now pick to alter either sample rate or bit depth
            proportion = len(self.sample_rates) / (
                len(self.sample_rates) + len(self.bit_depths)
            )
            if random.random() < proportion:
                # Random sample rate
                sample_rate = random.choice(self.sample_rates)
                sample_b = next(
                    filter(
                        lambda a: a.sample_rate == sample_rate
                        and a.bit_depth == sample_a.bit_depth,
                        snippet_list,
                    )
                )
            else:
                # Random bit depth
                bit_depth = random.choice(self.bit_depths)
                sample_b = next(
                    filter(
                        lambda a: a.bit_depth == bit_depth
                        and a.sample_rate == sample_a.sample_rate,
                        snippet_list,
                    )
                )
            test = Test(file, sample_a, sample_b)
            self.tests.append(test)

    def prepare(self):
        self.prepare_audio()
        self.prepare_tests()

        def finish_prep(dt):
            self.update_test_progress()
            load_pb = self.screen_manager.get_screen("intro").ids.load_pb
            load_pb.value = load_pb.max
            start_btn = self.screen_manager.get_screen("intro").ids.start_btn
            start_btn.text = "Begin Test"
            start_btn.disabled = False

        Clock.schedule_once(finish_prep)

    def prefer(self, id):
        self.tests[self.cur_test].result = id

        if DEBUG:
            for test_id, test in enumerate(self.tests):
                print(test_id, str(test))
            print("scores:", self.get_results()[1:])

        self.next()

    def prev(self):
        if self.cur_test > 0:
            self.cur_test -= 1
            self.update_test_progress()

    def next(self):
        if self.cur_test < self.num_tests - 1:
            self.cur_test += 1
            self.update_test_progress()
        else:
            self.show_results()

    def update_test_progress(self):
        test_pb = self.screen_manager.get_screen("test").ids.test_pb
        test_pb.value = min(
            test_pb.max, test_pb.max * ((self.cur_test + 1) / self.num_tests)
        )
        test_lbl = self.screen_manager.get_screen("test").ids.test_lbl
        test_lbl.text = f"{self.cur_test+1} / {self.num_tests}"

    def show_results(self):
        results, sample_rate_score, bit_depth_score, overall_score = self.get_results()
        text = ""
        test_colors = {
            True: "#00FF00",
            False: "FF0000",
            None: "FFFFFF",
        }
        for test_id, test in enumerate(self.tests):
            test_result = results[test_id]
            test_color = test_colors[test_result]
            test_text = f"[color={test_color}]{test_id+1}. {test}[/color]"
            if test_result is None:
                test_text = f"[s]{test_text}[/s]"
            test_text += "\n"
            text += test_text

        text += f"\n[b]Sample Rate Score: {round(100 * sample_rate_score, 2)}%[/b]\n"
        text += f"[b]Bit Depth Score: {round(100 * bit_depth_score, 2)}%[/b]\n"
        text += f"[b]Overall Score: {round(100 * overall_score, 2)}%[/b]"
        results_lbl = self.screen_manager.get_screen("results").ids.results_lbl
        results_lbl.text = text
        self.screen_manager.current = "results"

    def play(self, id):
        test = self.tests[self.cur_test]
        audio = test.sample_a if id == "A" else test.sample_b
        simpleaudio.play_buffer(
            audio.samples, 2, audio.bit_depth // 8, audio.sample_rate
        )

    def is_bit_depth_test(self, test) -> bool:
        a, b = test.sample_a, test.sample_b
        return a.bit_depth != b.bit_depth

    def check_bit_depth_test(self, test) -> bool:
        a, b, result = test.sample_a, test.sample_b, test.result
        return (a.bit_depth > b.bit_depth and result == "A") or (
            a.bit_depth < b.bit_depth and result == "B"
        )

    def is_sample_rate_test(self, test) -> bool:
        a, b = test.sample_a, test.sample_b
        return a.sample_rate != b.sample_rate

    def check_sample_rate_test(self, test) -> bool:
        a, b, result = test.sample_a, test.sample_b, test.result
        return (a.sample_rate > b.sample_rate and result == "A") or (
            a.sample_rate < b.sample_rate and result == "B"
        )

    def get_results(self) -> float:
        results = []
        sample_rate_tests_correct = 0
        sample_rate_tests = 0
        bit_depth_tests_correct = 0
        bit_depth_tests = 0
        neither_correct = 0
        total = 0
        for test in self.tests:
            if test.result is None:
                results.append(None)
            else:
                total += 1
                if self.is_sample_rate_test(test):
                    sample_rate_tests += 1
                    results.append(self.check_sample_rate_test(test))
                    sample_rate_tests_correct += int(results[-1])
                elif self.is_bit_depth_test(test):
                    bit_depth_tests += 1
                    results.append(self.check_bit_depth_test(test))
                    bit_depth_tests_correct += int(results[-1])
                else:
                    results.append(test.result == "Neither")
                    neither_correct += int(results[-1])

        sample_rate_score = sample_rate_tests_correct / max(1, sample_rate_tests)
        bit_depth_score = bit_depth_tests_correct / max(1, bit_depth_tests)
        overall_score = (
            sample_rate_tests_correct + bit_depth_tests_correct + neither_correct
        ) / max(1, total)

        return results, sample_rate_score, bit_depth_score, overall_score

    def publish(self):
        tests = [f"{test}\n" for test in self.tests]
        results = f"{self.get_results()[1:]}\n"
        with open("results.txt", "a+t") as fp:
            fp.write(f"seed: {SEED}\n")
            fp.writelines(tests)
            fp.write(results)

    def restart(self):
        self.prepare_tests()
        self.cur_test = 0
        self.update_test_progress()
        self.screen_manager.current = "test"

    def build(self):
        sm = ScreenManager()
        self.screen_manager = sm
        sm.add_widget(IntroScreen())
        sm.add_widget(TestScreen())
        sm.add_widget(ResultsScreen())
        threading.Thread(target=self.prepare).start()
        return sm


if __name__ == "__main__":
    if DEBUG:
        print("seed:", SEED)
    random.seed(SEED)
    SnakeOilApp().run()
