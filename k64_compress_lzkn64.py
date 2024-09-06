#!/usr/bin/env python3

"""
Script for recompressing a decompressed baserom file with zlib-compressed Nisitenma-Ichigo files.

Modified from the script that was in Fluvian's MNSG decompilation (before he mysteriously vanished).
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
def compress_buffer(file_buffer, pad_zeroes=False):
    """
    Compresses the data in the buffer specified in the arguments.
    :param file_buffer:
    :param pad_zeroes:
    :return:
    """
    # Size of the buffer to compress
    buffer_size = len(file_buffer)

    # Position of the current read location in the buffer.
    buffer_position = 0

    # Position of the current write location in the written buffer.
    write_position = 4

    # Allocate write_buffer with size of 0xFFFFFF (24-bit).
    write_buffer = np.zeros(MAX_NI_FILE_SIZE, dtype=np.uint8)

    # Position in the input buffer of the last time one of the copy modes was used.
    buffer_last_copy_position = 0

    while buffer_position < buffer_size:
        # Calculate maximum length we are able to copy without going out of bounds.
        if COPY_SIZE <= buffer_size - buffer_position:
            sliding_window_maximum_length = COPY_SIZE
        else:
            sliding_window_maximum_length = buffer_size - buffer_position

        # Calculate how far we are able to look back without going behind the start of the uncompressed buffer.
        if buffer_position - WINDOW_SIZE > 0:
            sliding_window_maximum_offset = buffer_position - WINDOW_SIZE
        else:
            sliding_window_maximum_offset = 0

        # Calculate maximum length the forwarding looking window is able to search.
        if RLE_SIZE < buffer_size - buffer_position - 1:
            forward_window_maximum_length = RLE_SIZE
        else:
            forward_window_maximum_length = buffer_size - buffer_position

        if forward_window_maximum_length > COPY_SIZE:
            for i in range(COPY_SIZE + 1, forward_window_maximum_length + 1):
                if (i + buffer_position) & 0xFFF in [0x021, 0x421, 0x821, 0xC21]:
                    forward_window_maximum_length = i
                    break

        sliding_window_match_position = -1
        sliding_window_match_size = 0

        forward_window_match_value = 0
        forward_window_match_size = 0

        # The current mode the compression algorithm prefers. (0x7F == None)
        current_mode = MODE_NONE

        # The current submode the compression algorithm prefers.
        current_submode = MODE_NONE

        # How many bytes will have to be copied in the raw copy command.
        raw_copy_size = buffer_position - buffer_last_copy_position

        # How many bytes we still have to copy in RLE matches with more than 0x21 bytes.
        rle_bytes_left = 0

        """Go backwards in the buffer, is there a matching value?
        If yes, search forward and check for more matching values in a loop.
        If no, go further back and repeat."""
        for search_position in range(
            buffer_position - 1, sliding_window_maximum_offset - 1, -1
        ):
            matching_sequence_size = 0

            while (
                file_buffer[search_position + matching_sequence_size]
                == file_buffer[buffer_position + matching_sequence_size]
            ):
                matching_sequence_size += 1

                if matching_sequence_size >= sliding_window_maximum_length:
                    break

            # Once we find a match or a match that is bigger than the match before it, we save its position and length.
            if matching_sequence_size > sliding_window_match_size:
                sliding_window_match_position = search_position
                sliding_window_match_size = matching_sequence_size

        """Look one step forward in the buffer, is there a matching value?
        If yes, search further and check for a repeating value in a loop.
        If no, continue to the rest of the function."""
        matching_sequence_value = file_buffer[buffer_position]
        matching_sequence_size = 0

        # If our matching sequence number is not 0x00, set the forward window maximum length to the copy size minus 1.
        # This is the highest it can really be in that case.
        if (
            matching_sequence_value != 0
            and forward_window_maximum_length > COPY_SIZE - 1
        ):
            forward_window_maximum_length = COPY_SIZE - 1

        while (
            file_buffer[buffer_position + matching_sequence_size]
            == matching_sequence_value
        ):
            matching_sequence_size += 1

            # If we find a sequence of matching values, save them.
            if matching_sequence_size >= 1:
                forward_window_match_value = matching_sequence_value
                forward_window_match_size = matching_sequence_size

            if matching_sequence_size >= forward_window_maximum_length:
                break

        # Try to pick which mode works best with the current values.
        if (
            sliding_window_match_size >= 4
            and sliding_window_match_size > forward_window_match_size
        ):
            current_mode = MODE_WINDOW_COPY

        elif forward_window_match_size >= 3:
            current_mode = MODE_RLE_WRITE_A

            if forward_window_match_value != 0x00:
                current_submode = MODE_RLE_WRITE_A
                rle_bytes_left = forward_window_match_size
            elif (
                forward_window_match_value == 0x00
                and forward_window_match_size < COPY_SIZE
            ):
                current_submode = MODE_RLE_WRITE_B
            elif (
                forward_window_match_value == 0x00
                and forward_window_match_size >= COPY_SIZE
            ):
                current_submode = MODE_RLE_WRITE_C

        elif forward_window_match_size >= 2 and forward_window_match_value == 0x00:
            current_mode = MODE_RLE_WRITE_A
            current_submode = MODE_RLE_WRITE_B

        """Write a raw copy command when these following conditions are met:
        The current mode is set and there are raw bytes available to be copied.
        The raw byte length exceeds the maximum length that can be stored.
        Raw bytes need to be written due to the proximity to the end of the buffer."""
        if (
            (current_mode != MODE_NONE and raw_copy_size >= 1)
            or raw_copy_size >= 0x1F
            or (buffer_position + 1) == buffer_size
        ):
            if buffer_position + 1 == buffer_size:
                raw_copy_size = buffer_size - buffer_last_copy_position

            while raw_copy_size > 0:
                if raw_copy_size > 0x1F:
                    write_buffer[write_position] = MODE_RAW_COPY | 0x1F
                    write_position += 1

                    for written_bytes in range(0x1F):
                        write_buffer[write_position] = file_buffer[
                            buffer_last_copy_position
                        ]
                        write_position += 1
                        buffer_last_copy_position += 1

                    raw_copy_size -= 0x1F
                else:
                    write_buffer[write_position] = MODE_RAW_COPY | raw_copy_size & 0x1F
                    write_position += 1

                    for written_bytes in range(raw_copy_size):
                        write_buffer[write_position] = file_buffer[
                            buffer_last_copy_position
                        ]
                        write_position += 1
                        buffer_last_copy_position += 1

                    raw_copy_size = 0

        if current_mode == MODE_WINDOW_COPY:
            write_buffer[write_position] = (
                MODE_WINDOW_COPY
                | ((sliding_window_match_size - 2) & 0x1F) << 2
                | (((buffer_position - sliding_window_match_position) & 0x300) >> 8)
            )
            write_position += 1
            write_buffer[write_position] = (
                buffer_position - sliding_window_match_position
            ) & 0xFF
            write_position += 1

            buffer_position += sliding_window_match_size
            buffer_last_copy_position = buffer_position

        elif current_mode == MODE_RLE_WRITE_A:
            if current_submode == MODE_RLE_WRITE_A:
                write_buffer[write_position] = (
                    MODE_RLE_WRITE_A | (forward_window_match_size - 2) & 0x1F
                )
                write_position += 1
                write_buffer[write_position] = forward_window_match_value & 0xFF
                write_position += 1

            elif current_submode == MODE_RLE_WRITE_B:
                write_buffer[write_position] = (
                    MODE_RLE_WRITE_B | (forward_window_match_size - 2) & 0x1F
                )
                write_position += 1

            elif current_submode == MODE_RLE_WRITE_C:
                write_buffer[write_position] = MODE_RLE_WRITE_C
                write_position += 1
                write_buffer[write_position] = (forward_window_match_size - 2) & 0xFF
                write_position += 1

            buffer_position += forward_window_match_size
            buffer_last_copy_position = buffer_position
        else:
            buffer_position += 1

    # Write the compressed size.
    write_buffer[1] = 0x00
    write_buffer[1] = write_position >> 16 & 0xFF
    write_buffer[2] = write_position >> 8 & 0xFF
    write_buffer[3] = write_position & 0xFF

    # Return the compressed buffer.
    if pad_zeroes:
        while write_position % 16 != 0:
            write_position += 1

    if write_position % 2 != 0:
        write_position += 1

    return write_buffer[0:write_position]


# Address of the first file in the overlay table.
first_file_addr = None

# Sizes of all the decompressed files combined.
raw_size = None

# Sizes of all the compressed files combined plus the ROM's main buffer size.
new_raw_size = None

# List of files to skip.
skip_files = []

# List of all addresses for the files.
file_addrs = []

# List of all file sizes for the files.
file_sizes = []


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


def get_decompressed_file_addresses_and_sizes(input, table_addr: int):
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

        file_addrs.append(file_addr)

        # Highest bit of address is set, file is already compressed.
        if input[table_addr + pos] == 0:
            skip_files.append(0)

            if (next_file_addr - file_addr) > 0:
                file_sizes.append(next_file_addr - file_addr)
            else:
                file_sizes.append(0)
        else:
            skip_files.append(1)

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
    # Max file size for a Nisitenma-Ichigo file.
    decompressed_buf = bytearray(MAX_NI_FILE_SIZE)

    for i in range(len(file_sizes)):
        copy_buffer_from_pos_with_len(
            input, decompressed_buf, file_addrs[i], file_sizes[i]
        )


def get_raw_file_addresses():
    pos = first_file_addr

    for i in range(len(file_addrs)):
        pos += file_sizes[i]

    global raw_size
    raw_size = pos - first_file_addr


def write_raw_files(input, buffer, table_addr):
    # Max file size for a compressed Nisitenma-Ichigo LZKN64 file.
    file_buf = bytearray(MAX_NI_FILE_SIZE)
    pos = first_file_addr

    print("New file addresses:")
    for i in range(0, len(file_addrs), 2):
        copy_buffer_from_pos_with_len(input, file_buf, file_addrs[i], file_sizes[i])

        if skip_files[i] != 1:
            file_buf = bytearray(compress_buffer(file_buf[0:file_sizes[i]]))

        copy_buffer_to_pos_with_len(
            file_buf, buffer, pos, len(file_buf)
        )

        # Write the new locations to the overlay table.
        buffer[table_addr + (i * 4) : table_addr + (i * 4) + 4] = (0x80000000 + pos).to_bytes(4, "big")
        buffer[table_addr + (i * 4) + 4: table_addr + (i * 4) + 8] = (len(file_buf) + pos).to_bytes(4, "big")

        # Only log unique file addresses
        if i > 0 and (pos - last_pos) > 0:
            print(f"{hex(pos)}")

        last_pos = pos
        pos += len(file_buf)

    global new_raw_size
    new_raw_size = pos


# Find the nearest power of two for the final ROM size.
# (https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2)


def get_new_file_size(size):
    new_size = 0x400000  # Smallest size of an N64 cartridge
    while new_size < size:
        new_size += 0x400000

    return new_size


def compress(input: bytearray, table_addr: int) -> bytearray:
    buffer = bytearray(MAX_ROM_SIZE)  # 512Mbit (64Mbyte) is the maximum ROM size.
    buffer = copy_buffer(input, buffer)

    # List all the file addresses and sizes in a table.
    get_decompressed_file_addresses_and_sizes(input, table_addr)

    # Get the decompressed file sizes.
    get_raw_file_sizes(input)

    # Get the decompressed file addresses.
    get_raw_file_addresses()

    buffer = zero_out_buffer_from_pos_with_len(buffer, first_file_addr, raw_size)

    write_raw_files(input, buffer, table_addr)

    return buffer[: get_new_file_size(new_raw_size)]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Nisitenma-Ichigo LZKN64 ROM compression Script", file=sys.stderr)
        print("", file=sys.stderr)
        print("compress.py input_file output_file", file=sys.stderr)
        print(
            "    input_file: Path to the ROM file for a decompressed Nisitenma-Ichigo LZKN64 game.",
            file=sys.stderr,
        )
        print(
            "    output_file: Path to the resulting compressed ROM file.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        with open(sys.argv[1], "rb") as input_file:
            input_buf = bytearray(input_file.read())

        table_addr = find_nisitenma_ichigo_offset(input_buf)
        if table_addr is None:
            print(
                "File must be a valid ROM containing a Nisitenma-Ichigo table with decompressed file offsets!",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(sys.argv[2], "wb") as output_file:
            output_file.write(compress(input_buf, table_addr))
