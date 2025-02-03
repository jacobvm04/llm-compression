from llm_compression import encode, decode

def test_round_trip_common_text():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    seed, ranks = encode(text)
    decoded_text = decode(seed, ranks)
    assert text == decoded_text
