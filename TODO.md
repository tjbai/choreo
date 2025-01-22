## core
- [ ] comprehensive benchmarking tests

## performance
- [ ] profile to identify hotspots and bottlenecks
- [ ] avoid materializing full attention mask for both versions
- [ ] flex attention score_mod
- [ ] flex attention block_mod

## parity
- [ ] training harness for ToT workflow
	- [ ] add teacher forcing to step or some parallel implementation
	- [ ] implement lora module
	- [ ] write main training loop with online/offline data collection

## ideas
- [ ] can we precompute block masks for the entire workflow?
