# SMLL Benchmark Results Summary

## Compression Ratio by Content Type

| Content Type | Original | SMLL | GZIP | ZSTD | Winner |
|--------------|----------|------|------|------|--------|
| Python Code | 1121 | 10.48x | 3.19x | 3.19x | SMLL |
| JavaScript Code | 1009 | 6.39x | 2.37x | 2.29x | SMLL |
| C Code | 884 | 11.19x | 2.60x | 2.53x | SMLL |
| Natural Prose | 1082 | 9.75x | 1.81x | 1.79x | SMLL |
| Wikipedia | 1320 | 14.83x | 2.05x | 2.02x | SMLL |
| JSON | 857 | 7.86x | 2.72x | 2.66x | SMLL |
| XML | 893 | 3.78x | 2.28x | 2.28x | SMLL |
| CSV | 512 | 4.20x | 1.74x | 1.67x | SMLL |
| LLM-Generated | 1197 | 14.96x | 1.89x | 1.86x | SMLL |
| Random Text | 500 | 1.27x | 1.22x | 1.19x | SMLL |
| UUIDs | 739 | 0.94x | 1.71x | 1.76x | ZSTD |
| Base64 | 400 | 1.26x | 1.19x | 1.17x | SMLL |
| Repetitive | 600 | 75.00x | 22.22x | 20.00x | SMLL |

## Key Findings

- **LLM-generated text**: SMLL excels at compressing text similar to what LLMs produce
- **Code**: High compression due to predictable syntax patterns
- **Random/High-entropy**: Traditional compressors perform better
- **Speed**: SMLL is slower but achieves competitive or better ratios on natural language
