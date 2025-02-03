## core

## performance
- [ ] profile to identify hotspots and bottlenecks
- [ ] avoid materializing full attention mask for both versions
- [ ] flex attention score_mod
- [ ] flex attention block_mod

## parity
- [ ] fuse lora weights for inference

## tot
- [ ] run training for longer and redo evals, particularly e2e performance
- [ ] finetune the baseline?

## mad
- [ ] setup bob/alice prisoners dilemma

## ideas
- [ ] can we precompute block masks for the entire workflow?
- [ ] computing rotation the fly for causal cross-attention
- [ ] explicit cache invalidation to force re-encoding
- [ ] lazy insert to minimize forward passes (register -> process?)

