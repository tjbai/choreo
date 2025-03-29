Tree of Thoughts (ToT) workflow that uses a three-phase approach to problem-solving. The workflow generates multiple solution candidates, selects the best through voting, and produces a final refined answer.

The specific workflow has these phases:
1. **Proposal Generation Phase**: The system generates 8 distinct solution proposals using chain-of-thought prompting.
2. **Voting Phase**: 4 independent voters evaluate all proposals and select the best candidate.
3. **Solution Refinement Phase**: The system produces a final answer based on the winning proposal.

Focus your evaluation on:
- Whether the generated proposals are diverse and correctly address the problem
- If voters properly evaluate proposals based on quality rather than superficial factors
- Whether the final refinement properly builds on the selected proposal
- Termination issues if no clear winner emerges from voting

Common failure modes include:
- FM-1.3: Proposals repeating the same reasoning paths
- FM-2.5: Voters ignoring substantive aspects of proposals
- FM-3.3: Incorrect verification in the voting process
- FM-2.6: Final solution that contradicts or ignores the winning proposal
