## core
- [ ] Implement parent_map as an adjacency matrix
- [ ] Add teacher forcing field to `step` for accurate benchmarking
  - [ ] Run comprehensive benchmarking tests
- [ ] Remove extraneous decoding in baseline interface
- [ ] Integrate and test spda
- [ ] Optimize stateless `step` calls by removing "top-off" forward passes

## performance
- [ ] flex attention score_mod
- [ ] flex attention block_mod

## parity
- [ ] training harness for ToT workflow

## ideas
- [ ] can we precompute block masks for the entire workflow?
