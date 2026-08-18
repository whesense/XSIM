from concurrent.futures import Future, ThreadPoolExecutor

import torch

from .image_grid import PNG_COMPRESSION_LEVEL, chw_byte_image, write_image


class AsyncImageWriter:
    """Encode and write PNGs on a thread pool, off the calling loop.

    torchvision's PNG encoder is a torch op and releases the GIL, so plain
    threads parallelise it: on a 1080x1920 render the throughput scales
    linearly to at least 16 threads. With enough of them to outrun whatever
    produces the images, saving costs the caller only the device-to-host copy.

    Use it as a context manager -- leaving the block drains the queue and
    re-raises whatever a worker hit, so a failed write is never silent::

        with AsyncImageWriter() as writer:
            for scene in ...:
                writer.write(path, render.result.color)

    Args:
        workers: encoder threads.
        compression_level: passed through to :func:`write_image`.
        queue_depth: images allowed in flight per worker. Each one holds a
            full-size host copy, so an unbounded queue is a leak whenever the
            producer is the faster side; ``write`` blocks instead.
    """

    def __init__(self, workers: int = 8,
                 compression_level: int = PNG_COMPRESSION_LEVEL,
                 queue_depth: int = 2):
        self.compression_level = compression_level
        self.max_pending = max(1, workers * queue_depth)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix='image_writer')
        self.pending: list[Future] = []

    def write(self, path: str, image: torch.Tensor):
        """Queue one image, in any layout :func:`write_image` accepts.

        The tensor is read before this returns, so the caller is free to drop
        or overwrite it -- and no device memory is pinned for the length of the
        queue.
        """
        image = chw_byte_image(image).cpu()

        self.drain(self.max_pending - 1)
        self.pending.append(self.pool.submit(
            write_image, path, image, self.compression_level))

    def drain(self, in_flight: int = 0):
        """Block until at most ``in_flight`` writes are still queued.

        Whatever exception a worker raised surfaces here, on the calling
        thread, since that is the one that can still do something about it.
        """
        still_queued = []
        for future in self.pending:
            if future.done():
                future.result()
            else:
                still_queued.append(future)
        self.pending = still_queued

        while len(self.pending) > in_flight:
            self.pending.pop(0).result()

    def close(self):
        self.drain()
        self.pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
