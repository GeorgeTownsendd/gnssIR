import struct
from pathlib import Path


def clean_ubx_line(line):
    """Clean a single line of UBX hex dump."""
    # Remove timestamp if present (e.g., "03:07:24")
    parts = line.split()
    if not parts:
        return ""

    # Find the hex part (after any timestamp/offset)
    hex_part = []
    for part in parts:
        # Keep only valid hex bytes
        if len(part) == 2 and all(c in '0123456789ABCDEFabcdef' for c in part):
            hex_part.append(part)

    return ' '.join(hex_part)


def read_ubx_file(filepath):
    """Read and clean UBX data from a file."""
    cleaned_hex = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned = clean_ubx_line(line)
            if cleaned:
                cleaned_hex.append(cleaned)

    # Join all hex strings and convert to bytes
    hex_str = ''.join(cleaned_hex).replace(' ', '')
    return bytes.fromhex(hex_str)


def decode_trkmeas(data):
    """Decode TRK-MEAS message following the C code structure."""
    if len(data) < 110:  # Minimum length check
        print("Message too short")
        return None

    # Verify sync chars
    if data[0:2] != bytes([0xB5, 0x62]):
        print("Invalid sync chars")
        return None

    # Basic message info
    msg = {
        'class': data[2],
        'id': data[3],
        'length': int.from_bytes(data[4:6], 'little'),
        'channels': []
    }

    # Get number of channels
    nch = data[8]  # At offset 6+2
    print(f"Number of channels: {nch}")

    # Process each channel (starting at offset 110)
    offset = 110
    for i in range(nch):
        if offset + 56 > len(data):
            print(f"Warning: Message truncated at channel {i}")
            break

        # Extract channel data following C code structure
        channel = {
            'quality': data[offset + 1],  # Quality indicator (0-7)
            'system': data[offset + 4],  # Navigation system
            'prn': data[offset + 5],  # Satellite number
            'freq': data[offset + 7] - 7,  # Frequency
            'flag': data[offset + 8],  # Tracking status
            'lock_code': data[offset + 16],  # Code lock count
            'lock_phase': data[offset + 17],  # Phase lock count
            'snr': struct.unpack('<H', data[offset + 20:offset + 22])[0] / 256.0,
            'ts': struct.unpack('<Q', data[offset + 24:offset + 32])[0],  # Transmission time
            'adr': struct.unpack('<Q', data[offset + 32:offset + 40])[0],  # Accumulated Doppler
            'dop': struct.unpack('<i', data[offset + 40:offset + 44])[0] * 0.0009765625  # Doppler (P2_10)
        }

        msg['channels'].append(channel)
        offset += 56

    return msg


def print_msg(msg):
    """Pretty print the decoded message."""
    print("\nDecoded TRK-MEAS Message:")
    print(f"Message Class: 0x{msg['class']:02X}")
    print(f"Message ID: 0x{msg['id']:02X}")
    print(f"Length: {msg['length']} bytes")
    print(f"\nChannels: {len(msg['channels'])}")

    for i, ch in enumerate(msg['channels']):
        print(f"\nChannel {i}:")
        print(f"  Quality: {ch['quality']}")
        print(f"  System: {ch['system']}")
        print(f"  PRN: {ch['prn']}")
        print(f"  Frequency: {ch['freq']}")
        print(f"  Tracking Flag: 0x{ch['flag']:02X}")
        print(f"  Lock Code/Phase: {ch['lock_code']}/{ch['lock_phase']}")
        print(f"  SNR: {ch['snr']:.1f}")
        print(f"  Transmission Time: {ch['ts']}")
        print(f"  Accumulated Doppler: {ch['adr']}")
        print(f"  Doppler: {ch['dop']:.3f}")


if __name__ == "__main__":
    filepath = Path("sec_message_new.txt")
    try:
        data = read_ubx_file(filepath)
        msg = decode_trkmeas(data)
        if msg:
            print_msg(msg)
    except Exception as e:
        print(f"Error processing file: {e}")