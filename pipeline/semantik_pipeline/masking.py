"""Brouillage léger (masque XOR mulberry32), DATA_CONTRACT §5."""
MASK32 = 0xFFFFFFFF


def _imul(left: int, right: int) -> int:
    return (left * right) & MASK32


def mulberry32(seed: int):
    """Générateur de uint32, identique à la référence JS du contrat."""
    state = seed & MASK32
    while True:
        state = (state + 0x6D2B79F5) & MASK32
        mixed = _imul(state ^ (state >> 15), 1 | state)
        mixed = ((mixed + _imul(mixed ^ (mixed >> 7), 61 | mixed)) & MASK32) ^ mixed
        yield (mixed ^ (mixed >> 14)) & MASK32


def keystream(seed: int, length: int) -> bytes:
    """`length` octets : chaque uint32 donne 4 octets, poids faible d'abord."""
    generator = mulberry32(seed)
    output = bytearray()
    while len(output) < length:
        output += next(generator).to_bytes(4, "little")
    return bytes(output[:length])


def xor_mask(data: bytes, seed: int) -> bytes:
    """Masque/démasque (opération symétrique), flux neuf à chaque appel."""
    return bytes(byte ^ key for byte, key in zip(data, keystream(seed, len(data))))
