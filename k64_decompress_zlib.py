#!/usr/bin/env python3

"""
Script for decompressing a baserom file with zlib-compressed Nisitenma-Ichigo files.

Modified from the script that was in Fluvian's MNSG decompilation (before he mysteriously vanished).
"""

import sys
import zlib
from typing import Optional

MAX_NI_FILE_SIZE = 0xFFFFFF
MAX_ROM_SIZE = 0x4000000
NISITENMA_ICHIGO_HEADER = (
    b"\x4E\x69\x73\x69\x74\x65\x6E\x6D\x61\x2D\x49\x63\x68\x69\x67\x6F"
)


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


def get_compressed_file_addresses_and_sizes(input, table_addr: int):
    pos = 0
    file_addr = int.from_bytes(
        input[table_addr + pos + 1: table_addr + pos + 4], byteorder="big"
    )
    next_file_addr = int.from_bytes(
        input[table_addr + pos + 5: table_addr + pos + 8], byteorder="big"
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
    # Max file size for a Nisitenma-Ichigo file.
    compressed_buf = bytearray(MAX_NI_FILE_SIZE)

    for i in range(len(file_sizes)):
        copy_buffer_from_pos_with_len(
            input, compressed_buf, file_addrs[i], file_sizes[i]
        )


def get_raw_file_addresses():
    pos = first_file_addr

    for i in range(len(file_addrs)):
        pos += file_sizes[i]

    global raw_size
    raw_size = pos - first_file_addr


def write_raw_files(input, buffer, table_addr):
    pos = first_file_addr

    print("New file addresses:")
    for i in range(0, len(file_addrs), 2):
        file_buf = input[file_addrs[i]: file_addrs[i] + file_sizes[i]]

        if skip_files[i] != 1:
            file_buf = bytearray(zlib.decompress(file_buf[4:]))

        copy_buffer_to_pos_with_len(
            file_buf, buffer, pos, len(file_buf)
        )

        # Write the new locations to the overlay table.
        buffer[table_addr + (i * 4) : table_addr + (i * 4) + 4] = pos.to_bytes(4, "big")
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

    return buffer[: get_new_file_size(new_raw_size)]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Nisitenma-Ichigo zlib ROM decompression Script", file=sys.stderr)
        print("", file=sys.stderr)
        print("decompress.py input_file output_file", file=sys.stderr)
        print(
            "    input_file: Path to the ROM file for a Nisitenma-Ichigo zlib-compressed game.",
            file=sys.stderr,
        )
        print(
            "    output_file: Path to the resulting decompressed ROM file.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        with open(sys.argv[1], "rb") as input_file:
            input_buf = bytearray(input_file.read())

        table_addr = find_nisitenma_ichigo_offset(input_buf)
        if table_addr is None:
            print(
                "File must be a valid ROM containing a Nisitenma-Ichigo table with compressed file offsets!",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(sys.argv[2], "wb") as output_file:
            output_file.write(decompress(input_buf, table_addr))
