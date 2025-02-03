from llm_compression import encode, decode
import bitstring
import lzma

if __name__ == '__main__':
    # from http://www.incompleteideas.net/IncIdeas/BitterLesson.html
    test_str = """The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning."""
    print("Original string:", test_str.split(" ", maxsplit=1)[1]) # splitting off the first word to remove the seed token

    # LLM Compression
    seed, encoded = encode(test_str)
    decoded = decode(seed, encoded)

    print("Decoded string:", decoded)
    print("Encoded:", encoded)
    print("Encoded length:", len(encoded.bin) / 8)

    # No Compression
    raw_bitstring = bitstring.BitArray(bytes(test_str.split(" ", maxsplit=1)[1], "utf-8"))
    print("Raw:", raw_bitstring)
    print("Raw length:", len(raw_bitstring.bin) / 8)
    print("Compression ratio:", len(raw_bitstring.bin) / len(encoded.bin))

    # LZMA Compression
    lzma_bitstring = bitstring.BitArray(lzma.compress(bytes(test_str.split(" ", maxsplit=1)[1], "utf-8"), preset=lzma.PRESET_EXTREME))
    print("LZMA:", lzma_bitstring)
    print("LZMA length:", len(lzma_bitstring.bin) / 8)
    print("LZMA compression ratio:", len(raw_bitstring.bin) / len(lzma_bitstring.bin))

