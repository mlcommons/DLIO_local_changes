"""
Arrow IPC reader using memory-mapped files for zero-copy access.

Opens Arrow IPC files via ``pa.memory_map()`` + ``pa.ipc.open_file()`` so the
OS page cache handles all I/O — no explicit read() syscalls, no data copies
into Python heap.
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ArrowReader(FormatReader):
    """
    Memory-mapped Arrow IPC reader.

    Uses ``pa.memory_map`` for true zero-copy reads — the kernel page cache
    serves data directly into the process address space.
    """
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        self.logger.info(
            f"{utcnow()} ArrowReader thread={thread_index} epoch={epoch}"
        )

    @dlp.log
    def open(self, filename):
        """Memory-map an Arrow IPC file and build cumulative row offsets."""
        import pyarrow as pa

        mmap_file = pa.memory_map(filename, 'r')
        reader = pa.ipc.open_file(mmap_file)

        # Build cumulative row offsets from record batches
        offsets = [0]
        for i in range(reader.num_record_batches):
            offsets.append(offsets[-1] + reader.get_batch(i).num_rows)

        return (mmap_file, reader, offsets)

    @dlp.log
    def close(self, filename):
        """Close the memory-mapped file."""
        if filename in self.open_file_map:
            entry = self.open_file_map[filename]
            if entry is not None:
                entry[0].close()
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """Read the record batch containing sample_index via zero-copy memory-mapped access."""
        mmap_file, reader, offsets = self.open_file_map[filename]

        # Binary search for the batch containing this sample
        batch_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        batch_idx = min(batch_idx, reader.num_record_batches - 1)

        batch = reader.get_batch(batch_idx)

        # Touch every page to trigger mmap page faults and ensure full I/O
        PAGE_SIZE = 4096
        for col in batch.columns:
            for buf in col.buffers():
                if buf is not None and buf.size > 0:
                    mv = memoryview(buf)
                    # Touch one byte per page to fault in entire buffer
                    for offset in range(0, len(mv), PAGE_SIZE):
                        _ = mv[offset]

        dlp.update(image_size=batch.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
