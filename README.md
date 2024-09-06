# k64-rom-decompressors
Python scripts for decompressing and recompressing Konami N64 ROMs that use Nisitenma-Ichigo table filesystems.

## Prerequisites
```sh
pip install numpy
```
```sh
pip install numba
```

## Usage
For their primary filesystem, Konami N64 games are known to use a table headered "Nisitenma-Ichigo" that contains pointers to the beginning and end offsets in the ROM of the external files that it loads. Most, if not all, of these files will be compressed in one of two formats: a proprietary one known as LZKN64, or the far more standard zlib. Once you've identified what compression format your game's files are using, place it and that format's k64_decompress script in the same directory and run said script with the following arguments to generate a version of the ROM that will have all of those files decompressed inside it:

`"input_file.z64"` `"output_file.z64"`

For the decompressed ROM to function properly, you may need to update the CRC afterwards. In more extreme cases (such as with Castlevania 64), further modifications to the ROM may be necessary. To compress the ROM back, run the k64_compress script instead, again with the same arguments as above.

## Credits
- Fluvian and [@LiquidCat64](https://github.com/LiquidCat64)
for reversing the [LZKN64](https://github.com/Fluvian/lzkn64) compression algorithm used by Konami
