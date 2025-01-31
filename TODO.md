## core

## performance
- [ ] profile to identify hotspots and bottlenecks
- [ ] avoid materializing full attention mask for both versions
- [ ] flex attention score_mod
- [ ] flex attention block_mod

## parity
- [ ] fuse lora weights for inference

## ideas
- [ ] can we precompute block masks for the entire workflow?
- [ ] computing rotation the fly for causal cross-attention
