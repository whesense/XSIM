import torch
from subprocess import Popen, PIPE


class FFMpegWriter:
    def __init__(
            self,
            out_path,
            fps: float = 25,
            codec: str = 'libx264',
            pix_fmt: str = 'yuv420p',
            crf: int = 5,
            threads: int = 24
    ):
        self.out_path = out_path
        self.command = (
                "ffmpeg -y -loglevel panic -f rawvideo -pix_fmt rgb24 -s {}x{}" +
                " -r {} -i - -c:v {} -pix_fmt {} -crf {} -preset slow -threads {} {}".format(
                fps, codec, pix_fmt, crf, threads, self.out_path))
        print(self.command)
        self.fps = fps
        self._p = None
        self._width = 0
        self._height = 0

    def __enter__(self):
        self._p = None
        self._width = 0
        self._height = 0
        return self

    def write(self, frame):
        h, w, c = frame.shape
        assert c == 3

        if self._p is None:
            self._height = h
            self._width = w
            self._p = Popen(self.command.format(w, h).split(), stdin=PIPE)
        else:
            assert self._height == h
            assert self._width == w

        if isinstance(frame, torch.Tensor):
            if frame.dtype == torch.float32:
                frame = frame.clamp(0, 1).mul(255).byte()
            if frame.device != torch.cpu:
                frame = frame.cpu()
            frame = frame.numpy()
        self._p.stdin.write(frame.tobytes())

    def __exit__(self, *args, **kwargs):
        if self._p is not None:
            self._p.stdin.close()
            self._p.wait()
