#!/usr/bin/env python3

"""
Script for decompressing a baserom file with lzkn64-compressed Nisitenma-Ichigo files.

Modified from the script that was in Fluvian's MNSG decompilation (before he mysteriously vanished)
to use @LiquidCat64's rework of Fluvian's LZKN64 compression.
"""

import sys
import numpy as np
from numba import njit
from typing import Optional

MAX_NI_FILE_SIZE = 0xFFFFFF
MAX_ROM_SIZE = 0x4000000
NISITENMA_ICHIGO_HEADER = (
    b"\x4E\x69\x73\x69\x74\x65\x6E\x6D\x61\x2D\x49\x63\x68\x69\x67\x6F"
)


TYPE_COMPRESS = 1
TYPE_DECOMPRESS = 2

MODE_NONE = 0x7F
MODE_WINDOW_COPY = 0x00
MODE_RAW_COPY = 0x80
MODE_RLE_WRITE_A = 0xC0
MODE_RLE_WRITE_B = 0xE0
MODE_RLE_WRITE_C = 0xFF

WINDOW_SIZE = 0x3DF
COPY_SIZE = 0x21
RLE_SIZE = 0x101


def find_nisitenma_ichigo_offset(rom_buffer: bytes) -> Optional[int]:
    """
    Find the byte-offset of a Nisitenma-Ichigo table
    :param rom_buffer:
    :return: the byte-offset at the end of the Nisitenma-Ichigo header if present, None otherwise
    """
    offset = rom_buffer.find(NISITENMA_ICHIGO_HEADER)
    if offset == -1:
        return None

    return offset + len(NISITENMA_ICHIGO_HEADER)

@njit
def decompress_buffer(file_buffer):
    """
    Decompresses the data in the buffer specified in the arguments.
    :param file_buffer:
    :return:
    """
    # Position of the current read location in the buffer.
    buffer_position = 4

    # Position of the current write location in the written buffer.
    write_position = 0

    # Get compressed size.
    compressed_size = (
        (file_buffer[1] << 16) + (file_buffer[2] << 8) + file_buffer[3] - 1
    )

    # Allocate write_buffer with size of 0xFFFFFF (24-bit).
    write_buffer = np.zeros(MAX_NI_FILE_SIZE, dtype=np.uint8)

    while buffer_position < compressed_size:
        mode_command = file_buffer[buffer_position]
        buffer_position += 1

        if MODE_WINDOW_COPY <= mode_command < MODE_RAW_COPY:
            copy_length = (mode_command >> 2) + 2
            copy_offset = file_buffer[buffer_position] + (mode_command << 8) & 0x3FF
            buffer_position += 1

            for current_length in range(copy_length, 0, -1):
                write_buffer[write_position] = write_buffer[
                    write_position - copy_offset
                ]
                write_position += 1
        elif MODE_RAW_COPY <= mode_command < MODE_RLE_WRITE_A:
            copy_length = mode_command & 0x1F

            for current_length in range(copy_length, 0, -1):
                write_buffer[write_position] = file_buffer[buffer_position]
                write_position += 1
                buffer_position += 1
        elif MODE_RLE_WRITE_A <= mode_command <= MODE_RLE_WRITE_C:
            write_length = 0
            write_value = 0x00

            if MODE_RLE_WRITE_A <= mode_command < MODE_RLE_WRITE_B:
                write_length = (mode_command & 0x1F) + 2
                write_value = file_buffer[buffer_position]
                buffer_position += 1
            elif MODE_RLE_WRITE_B <= mode_command < MODE_RLE_WRITE_C:
                write_length = (mode_command & 0x1F) + 2
            elif mode_command == MODE_RLE_WRITE_C:
                write_length = file_buffer[buffer_position] + 2
                buffer_position += 1

            for current_length in range(write_length, 0, -1):
                write_buffer[write_position] = write_value
                write_position += 1

    while write_position % 16 != 0:
        write_position += 1

    # Return the decompressed buffer.
    return write_buffer[0:write_position]


# Address of the first file in the overlay table.
first_file_addr = None

# Sizes of all the decompressed files combined.
raw_size = None

# List of files to skip.
skip_files = []

# List of all addresses for the files.
file_addrs = []

# List of all file sizes for the files.
file_sizes = []

# List of the new addresses for the files.
new_file_addrs = []

# List of all the decompressed file sizes.
new_file_sizes = []

##### Search #####


def decompress_files(input, size_compressed):
    buffer = bytearray(
        MAX_NI_FILE_SIZE
    )  # Max file size for a compressed Nisitenma-Ichigo file.

    in_pos = 4  # Offset in input file.
    buf_pos = 0  # Offset in output file.

    while in_pos < size_compressed:
        cur_cmd = input[in_pos]
        in_pos += 1

        if cur_cmd < 0x80:  # Sliding window lookback and copy with length.
            look_back_length = input[in_pos] + (cur_cmd << 8) & 0x3FF
            for _ in range(2 + (cur_cmd >> 2)):
                buffer[buf_pos] = buffer[buf_pos - look_back_length]
                buf_pos += 1
            in_pos += 1

        elif cur_cmd < 0xA0:  # Raw data copy with length.
            for _ in range(cur_cmd & 0x1F):
                buffer[buf_pos] = input[in_pos]
                buf_pos += 1
                in_pos += 1

        elif cur_cmd <= 0xFF:  # Write specific byte for length.
            value = 0
            length = 2 + (cur_cmd & 0x1F)

            if cur_cmd == 0xFF:
                length = 2 + input[in_pos]
                in_pos += 1
            elif cur_cmd < 0xE0:
                value = input[in_pos]
                in_pos += 1

            for _ in range(length):
                buffer[buf_pos] = value
                buf_pos += 1

        else:
            in_pos += 1

    return buffer[:buf_pos]


# Decompression code modified to just increment the position counters.


def decompress_get_len(input, size_compressed) -> int:
    in_pos = 4  # Offset in input file.
    buf_pos = 0  # Offset in output file.

    while in_pos < size_compressed:
        cur_cmd = input[in_pos]
        in_pos += 1

        if cur_cmd < 0x80:  # Sliding window lookback and copy with length.
            for _ in range(2 + (cur_cmd >> 2)):
                buf_pos += 1
            in_pos += 1

        elif cur_cmd < 0xA0:  # Raw data copy with length.
            for _ in range(cur_cmd & 0x1F):
                buf_pos += 1
                in_pos += 1

        elif cur_cmd <= 0xFF:  # Write specific byte for length.
            length = 2 + (cur_cmd & 0x1F)

            if cur_cmd == 0xFF:
                length = 2 + input[in_pos]
                in_pos += 1
            elif cur_cmd < 0xE0:
                in_pos += 1

            for _ in range(length):
                buf_pos += 1

        else:
            in_pos += 1

    return buf_pos


def copy_buffer(input, output: bytearray) -> bytearray:
    output[0 : len(input)] = input

    return output


def copy_buffer_from_pos_with_len(
    input: bytearray, output: bytearray, pos: int, len: int
) -> bytearray:
    output[0:len] = input[pos : pos + len]

    return output


def copy_buffer_to_pos_with_len(
    input: bytearray, output: bytearray, pos: int, len: int
) -> bytearray:
    output[pos : pos + len] = input[0:len]

    return output


def zero_out_buffer_from_pos_with_len(
    output: bytearray, pos: int, len: int
) -> bytearray:
    for i in range(len):
        output[i + pos] = 0

    return output


def get_compressed_file_addresses_and_sizes(input, table_addr: int):
    pos = 0
    file_addr = int.from_bytes(
        input[table_addr + pos + 1 : table_addr + pos + 4], byteorder="big"
    )
    next_file_addr = int.from_bytes(
        input[table_addr + pos + 5 : table_addr + pos + 8], byteorder="big"
    )

    global first_file_addr
    first_file_addr = file_addr

    while file_addr != 0:
        # Highest bit of address is not set, file is already decompressed.
        if input[table_addr + pos] == 0:
            skip_files.append(1)

            file_addrs.append(file_addr)

            if (next_file_addr - file_addr) > 0:
                file_sizes.append(next_file_addr - file_addr)
            else:
                file_sizes.append(0)
        else:
            skip_files.append(0)

            file_addrs.append(file_addr)

            # Headers of compressed files have their compressed sizes within them.
            file_sizes.append(
                int.from_bytes(input[file_addr + 1 : file_addr + 4], byteorder="big")
            )

        pos += 4

        file_addr = int.from_bytes(
            input[table_addr + pos + 1 : table_addr + pos + 4], byteorder="big"
        )
        next_file_addr = int.from_bytes(
            input[table_addr + pos + 5 : table_addr + pos + 8], byteorder="big"
        )


def get_raw_file_sizes(input):
    # Max file size for a compressed Nisitenma-Ichigo file.
    compressed_buf = bytearray(MAX_NI_FILE_SIZE)

    for i in range(len(file_sizes)):
        copy_buffer_from_pos_with_len(
            input, compressed_buf, file_addrs[i], file_sizes[i]
        )

        if skip_files[i] != 1:
            # "Fake decompress" to get the length of the raw data.
            new_file_sizes.append(decompress_get_len(compressed_buf, file_sizes[i]))
        else:
            new_file_sizes.append(file_sizes[i])


def get_raw_file_addresses():
    pos = first_file_addr

    for i in range(len(file_addrs)):
        new_file_addrs.append(pos)
        pos += new_file_sizes[i]

    global raw_size
    raw_size = pos - first_file_addr


def write_raw_files(input, buffer, table_addr):
    # Max file size for a compressed Nisitenma-Ichigo file.
    file_buf = bytearray(MAX_NI_FILE_SIZE)

    print("New file addresses:")
    for i in range(len(file_addrs)):
        copy_buffer_from_pos_with_len(input, file_buf, file_addrs[i], file_sizes[i])

        if skip_files[i] != 1:
            file_buf = bytearray(decompress_buffer(file_buf))

        copy_buffer_to_pos_with_len(
            file_buf, buffer, new_file_addrs[i], new_file_sizes[i]
        )

        # Write the new locations to the overlay table.
        buffer[table_addr + (i * 4) : table_addr + (i * 4) + 4] = new_file_addrs[i].to_bytes(4, "big")

        # Only log unique file addresses
        if i > 0 and (new_file_addrs[i] - new_file_addrs[i - 1]) > 0:
            print(f"{hex(new_file_addrs[i])}")


# Find the nearest power of two for the final ROM size.
# (https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2)


def get_new_file_size(size):
    size -= 1
    size |= size >> 1
    size |= size >> 2
    size |= size >> 4
    size |= size >> 8
    size |= size >> 16
    size += 1

    return size


def decompress(input: bytearray, table_addr: int) -> bytearray:
    buffer = bytearray(MAX_ROM_SIZE)  # 512Mbit (64Mbyte) is the maximum ROM size.
    buffer = copy_buffer(input, buffer)

    # List all the file addresses and sizes in a table.
    get_compressed_file_addresses_and_sizes(input, table_addr)

    # Get the decompressed file sizes.
    get_raw_file_sizes(input)

    # Get the decompressed file addresses.
    get_raw_file_addresses()

    buffer = zero_out_buffer_from_pos_with_len(buffer, first_file_addr, raw_size)

    write_raw_files(input, buffer, table_addr)

    return buffer[: get_new_file_size(raw_size + first_file_addr)]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Nisitenma-Ichigo LZKN64 ROM decompression Script", file=sys.stderr)
        print("", file=sys.stderr)
        print("decompress.py input_file output_file", file=sys.stderr)
        print(
            "    input_file: Path to the ROM file for a Nisitenma-Ichigo LZKN64-compressed game.",
            file=sys.stderr,
        )
        print(
            "    output_file: Path to the resulting decompressed ROM file.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        with open(sys.argv[1], "rb") as input_file:
            input_buf = input_file.read()

        table_addr = find_nisitenma_ichigo_offset(input_buf)
        if table_addr is None:
            print(
                "File must be a valid ROM containing a Nisitenma-Ichigo table with compressed file offsets!",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(sys.argv[2], "wb") as output_file:
            output_file.write(decompress(input_buf, table_addr))
