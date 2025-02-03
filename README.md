# LLM Compression

This project implements an entropy encoding algorithm for lossless text compression that uses an LLM as the source distribution.

The aim is to be a proof of concept, so I've not optimized it for speed yet. Right now it offers a very high compression ratio at the cost of very compute-intensive encoding and decoding.

Credit to [ts_sms](https://bellard.org/ts_sms/) for inspiration on the encoding scheme.

## Usage

```python
from llm_compression import encode, decode

seed, encoded = encode("Hello, world!")
decoded = decode(seed, encoded)

print(decoded)
# Hello, world!
```

## Example

```bash
python example.py
```

### Reference output

M4 Max Macbook Pro

```
> python example.py
Original string: biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
Computing probabilities: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 599/599 [02:09<00:00,  4.64it/s]
Encoding tokens: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 599/599 [05:20<00:00,  1.87it/s]
Decoding tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2745/2745 [08:32<00:00,  5.35it/s]
Decoded string: The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
Encoded: 0xa3e756cd793f4fd4b8df12b429e83c742c99285c9e07d8dd2b3fb8ba32da7a88e033d31267661b5d9612747864c6d50fa77bc2d908a0ae3ca67e6ae652bdb5677ff24947f97e5e7ba1f30d6725f2315fc890df13650f536f71412e7bd022ae31cea3be43fa652137ef384405a396096517bfff5db0154468bbcd193022...
Encoded length: 343.125
Raw: 0x62696767657374206c6573736f6e20746861742063616e20626520726561642066726f6d203730207965617273206f6620414920726573656172636820697320746861742067656e6572616c206d6574686f64732074686174206c6576657261676520636f6d7075746174696f6e2061726520756c74696d6174656c79...
Raw length: 3112.0
Compression ratio: 9.069581056466303
LZMA: 0xfd377a585a000004e6d6b446020021010c0000008f98419ce00c27059a5d00311a492d1eadee3ed176dfb5302d5456dc099e0edd4338205bd324dfade387ed26e783a7540785324977759320e951b88504ec038ab1ca311bcd9a2bb3bb6d91941670b6b88b4b1ffd6185681f4311f2b6c3ffd4dada01e72398172b5fc0...
LZMA length: 1500.0
LZMA compression ratio: 2.074666666666667
```
