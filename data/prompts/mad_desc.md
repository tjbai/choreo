Multi-agent iterative debate workflow involving three agents: Affirmative, Negative, and Moderator. The debate addresses mathematical reasoning problems through structured exchanges:

1. **Initialization Phase**:
   - Affirmative agent presents initial solution to the problem
   - Negative agent provides a counter-solution

2. **Debate Phase**: For each round (up to max_rounds):
   - Affirmative responds to Negative's previous argument
   - Negative responds to Affirmative's latest argument
   - Moderator evaluates both positions and determines if there's a clear winner

3. **Termination Phase**:
   - Debate concludes when Moderator indicates a preference ("Yes") with supporting reasoning
   - If max_rounds reached without decision, system forces final evaluation

4. **Output Structure**:
   - Moderator outputs decisions in JSON format with fields for Preference, Supported Side, Reason, and Answer

Focus your evaluation on:
- Whether debaters properly build upon or refute previous arguments
- If the moderator maintains objectivity while making evidence-based evaluations
- Whether termination decisions are justified by the debate content
- JSON parsing and format adherence issues that may affect workflow

Common failure modes include:
- FM-2.5: Debaters ignoring key counterarguments from previous rounds
- FM-1.2: Moderator showing bias rather than evidence-based evaluation
- FM-2.6: Moderator's reasoning not matching their preference decision
- FM-3.1: Premature termination before key issues are resolved
- FM-2.1: Debaters restarting arguments without addressing prior points
