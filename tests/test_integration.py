import torch
from unittest import TestCase
from typing import List
from operator import itemgetter as get

from llama.workflow import Workflow, Prompt, Task
from llama.model import Transformer
from llama.tokenizer import ChatFormat, Tokenizer

class TestWorkflowIntegration(TestCase):
    def setUp(self):
        class MockTransformer(Transformer):
            training = False

            def __init__(self):
                self.kv_cache = {}
                self.params = type("Params", (), {"max_seq_len": 2048})
                self.call_history = []

            def forward(self, tokens, start_pos, mask=None, position_ids=None):
                self.call_history.append({
                    "tokens": tokens.clone(),
                    "start_pos": start_pos,
                    "mask": mask.clone() if mask is not None else None,
                    "position_ids": position_ids.clone() if position_ids is not None else None
                })
                return torch.full((1, tokens.shape[1], 5), 0.1)

            def reposition_cache(self, where, from_pos, to_pos):
                self.call_history.append({
                    "operation": "reposition",
                    "where": where.clone(),
                    "from_pos": from_pos.clone(),
                    "to_pos": to_pos.clone()
                })

        class MockFormatter(ChatFormat):
            def __init__(self):
                ...

            def encode_message(self, message):
                return [2, 0, 2, 1, 0, 1]

            def encode_dialog(self, dialog):
                tokens = []
                for message in dialog:
                    tokens.extend(self.encode_message(message))
                return tokens

            def encode_header(self, message):
                return [2, 0, 2]

        class MockTokenizer(Tokenizer):
            def __init__(self):
                ...

            def encode(self, *_, **__):
                return [1]

            bos_id = 128000
            stop_tokens = [128001, 128009]

        self.model = MockTransformer()
        self.tokenizer = MockTokenizer()
        self.workflow = Workflow(
            self.model,
            self.tokenizer,
            max_nodes=10,
        )
        self.workflow.formatter = MockFormatter()

    def test_insert(self):
        prompts: List[Prompt] = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"}
                ],
                "parent_ids": []
            }
        ]
        self.workflow.insert(prompts)
        self.assertEqual(self.model.call_history[-1]["start_pos"], 1)
        self.assertEqual(len(self.model.call_history), 2)

        N = self.workflow.cache_len
        self.assertEqual(N, 13)
        self.assertEqual(
            self.workflow.context[:N].tolist(),
            [128000] + 2 * [2, 0, 2, 1, 0, 1]
        )
        self.assertEqual(
            self.workflow.position_map[:N].tolist(),
            list(range(13))
        )
        self.assertEqual(
            self.workflow.node_map[:N].tolist(),
            [0] + [1 for _ in range(12)]
        )
        self.assertEqual(
            self.workflow.adj[1, :5].tolist(),
            [True, True] + [False for _ in range(3)]
        )

    def test_parallel_insert(self):
        system_prompts: List[Prompt] = [
            {
                "messages": [{"role": "system", "content": ""}],
                "parent_ids": []
            },
            {
                "messages": [{"role": "system", "content": ""}],
                "parent_ids": []
            }
        ]

        [sys1, sys2] = self.workflow.insert(system_prompts)
        self.assertEqual(self.model.call_history[-1]["start_pos"], 1)
        self.assertEqual(
            self.model.call_history[-1]["position_ids"].tolist(),
            2 * [1, 2, 3, 4, 5, 6]
        )

        user_prompts: List[Prompt] = [
            {
                "messages": [{"role": "user", "content": ""}],
                "parent_ids": [sys1["id"]]
            },
            {
                "messages": [{"role": "user", "content": ""}],
                "parent_ids": [sys2["id"]]
            },
            {
                "messages": [{"role": "user", "content": ""}],
                "parent_ids": [sys1["id"], sys2["id"]]
            }
        ]

        [user1, user2, user3] = self.workflow.insert(user_prompts)

        self.assertEqual(self.model.call_history[-1]["start_pos"], 13)
        self.assertEqual(
            self.model.call_history[-1]["position_ids"].tolist(),
            3 * [7, 8, 9, 10, 11, 12]
        )
        self.assertEqual(self.model.call_history[-1]["mask"].shape, (18, 18 + 13))
        self.assertEqual(
            self.workflow.position_map[13:31].tolist(),
            3 * [7, 8, 9, 10, 11, 12]
        )
        self.assertEqual(self.workflow.cache_len, 31)

    def test_insert_and_step(self):
        prompts: List[Prompt] = [
            {
                "messages": [{"role": "system", "content": ""}],
                "parent_ids": []
            }
        ]
        [prompt] = self.workflow.insert(prompts)
        self.assertEqual(len(self.model.call_history), 2)

        tasks: List[Task] = [{
            "header": ("assistant", None),
            "parent_ids": [prompt["id"]],
            "prefill": None
        }]
        tokens, cached = get('tokens', 'nodes')(self.workflow.step(tasks, max_gen_len=10))

        # 1 for bos, 1 for step, 1 for prefill, 8 decoding, 1 for final decoding step, 1 top-off
        self.assertEqual(len(self.model.call_history), 13)

    def test_tot_e2e(self):
        from llama.workflows.tot import tot_cached

        def mock_decode(_self, tokens):
            return "BEST CHOICE: 1"
        self.tokenizer.decode = mock_decode.__get__(self.tokenizer, self.tokenizer.__class__)

        data = tot_cached(
            workflow=self.workflow,
            problem="What is 2 + 2?",
            branching_factor=2,
            voters=2
        )

        self.assertIn("proposal_tokens", data)
        self.assertIn("vote_tokens", data)
        self.assertIn("final_tokens", data)
        self.assertIn("votes", data)

        self.assertEqual(len(data["proposal_tokens"]), 2)
        self.assertEqual(len(data["vote_tokens"]), 2)
        self.assertEqual(data["votes"], [1, 1])
        self.assertIsNotNone(data["final_tokens"])

    def assert_parents_unmasked(self, call, tasks):
        """
        Helper to verify that each row in `call["mask"]` un-masks the tokens
        belonging to the node IDs in tasks[i]["parent_ids"].
        """
        mask = call["mask"]
        bsz = len(tasks)
        self.assertEqual(mask.shape[0], bsz)

        for i, task in enumerate(tasks):
            parent_ids = task["parent_ids"]
            all_parent_positions = []
            for pid in parent_ids:
                pos = (self.workflow.node_map[:self.workflow.cache_len] == pid).nonzero().flatten()
                all_parent_positions.append(pos)
            if not all_parent_positions:
                continue
            combined_positions = torch.cat(all_parent_positions)
            for p in combined_positions:
                val = float(mask[i, p].item())
                self.assertEqual(val, 0)

    def test_tot(self):
        # 1) Insert initial prompts
        [cot, vote, finish] = self.workflow.insert([
            {"messages": [{"role": "system", "content": ""}], "parent_ids": []},
            {"messages": [{"role": "system", "content": ""}], "parent_ids": []},
            {"messages": [{"role": "system", "content": ""}], "parent_ids": []},
        ])
        self.assertEqual((cot["id"], vote["id"], finish["id"]), (1, 2, 3))
        self.assertEqual(len(self.model.call_history), 2)
        self.assertEqual(self.model.call_history[-1]["tokens"].shape[1], sum(c["length"] for c in [cot, vote, finish]))
        self.assertEqual(self.model.call_history[-1]["mask"].shape, (18, 19))
        triu = torch.full((6, 6), float("-inf"))
        triu = torch.triu(triu, diagonal=1)
        self.assertTrue(torch.all(self.model.call_history[-1]["mask"][:6, 1:7] == triu))
        self.assertTrue(torch.all(self.model.call_history[-1]["mask"][6:12, 7:13] == triu))
        self.assertTrue(torch.all(self.model.call_history[-1]["mask"][12:, 13:] == triu))

        # 2) Decode parallel CoT branches
        BRANCHES = 2
        proposal_tasks: List[Task] = [
            {
                "header": ("assistant", f"proposal {i+1}"),
                "prefill": None,
                "parent_ids": [cot["id"]],
            }
            for i in range(BRANCHES)
        ]
        proposal_tokens, proposal_nodes = get('tokens', 'nodes')(self.workflow.step(proposal_tasks, max_gen_len=3))
        self.assertEqual([node["id"] for node in proposal_nodes], [4, 5])
        self.assertEqual(self.model.call_history[-1]["tokens"].shape[1], BRANCHES)
        self.assertEqual(self.model.call_history[-1]["mask"].shape[0], BRANCHES)
        self.assertTrue(torch.all(self.model.call_history[-1]["tokens"] == torch.tensor([128009, 128009]))) # force decode top-off
        self.assert_parents_unmasked(self.model.call_history[-1], proposal_tasks)

        # 3) Voters get to see prompt AND all branches
        VOTERS = 2
        voter_tasks: List[Task] = [
            {
                "header": ("assistant", None),
                "prefill": None,
                "parent_ids": [vote["id"]] + [p["id"] for p in proposal_nodes],
            }
            for _ in range(VOTERS)
        ]
        vote_tokens, vote_nodes = get('tokens', 'nodes')(self.workflow.step(voter_tasks, max_gen_len=3))
        self.assertEqual([vote["id"] for vote in vote_nodes], [6, 7])
        self.assertEqual(self.model.call_history[-1]["mask"].shape[0], VOTERS)
        self.assert_parents_unmasked(self.model.call_history[-1], voter_tasks)

        # 4) Final step sees prompt and best (last) proposal
        best_proposal_id = proposal_nodes[-1]["id"]
        final_task: List[Task] = [{
            "header": ("assistant", None),
            "prefill": None,
            "parent_ids": [finish["id"], best_proposal_id]
        }]
        final_tokens, final_nodes = get('tokens', 'nodes')(self.workflow.step(final_task, max_gen_len=3))
        self.assert_parents_unmasked(self.model.call_history[-1], final_task)
        self.assertTrue(torch.all(
            self.workflow.adj[:9, :9] ==
            torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0], # bos
                [1, 1, 0, 0, 0, 0, 0, 0, 0], # insert prompt
                [1, 0, 1, 0, 0, 0, 0, 0, 0], # vote prompt
                [1, 0, 0, 1, 0, 0, 0, 0, 0], # finish prmopt
                [1, 1, 0, 0, 1, 0, 0, 0, 0], # branch 1
                [1, 1, 0, 0, 0, 1, 0, 0, 0], # branch 2
                [1, 0, 1, 0, 1, 1, 1, 0, 0], # voter 1
                [1, 0, 1, 0, 1, 1, 0, 1, 0], # voter 2
                [1, 0, 0, 1, 0, 1, 0, 0, 1], # finish
            ], dtype=torch.bool)
        ))

    def test_teacher_force(self):
        tasks: List[Task] = [{"header": ("assistant", None), "parent_ids": []}]
        forced_tokens = torch.tensor([[101, 102, 128009]]).long()
        out_tokens, out_nodes = get('tokens', 'nodes')(self.workflow.step(tasks, max_gen_len=4, teacher_force=forced_tokens))
        self.assertEqual(out_tokens, [[101, 102]])
        self.assertEqual(len(self.model.call_history), 5)

        self.workflow.step(tasks, max_gen_len=4, stateless=True, teacher_force=forced_tokens)
        self.assertEqual(len(self.model.call_history), 5+3)

        forced_tokens = torch.tensor([[101, 102, 103]]).long()
        out_tokens, out_nodes = get('tokens', 'nodes')(self.workflow.step(tasks, max_gen_len=4, teacher_force=forced_tokens))
        self.assertEqual(out_tokens, forced_tokens.tolist())
        self.assertEqual(len(self.model.call_history), 5+3+5)

    def test_teacher_force_parallel(self):
        prompts: List[Prompt] = [{"messages": [{"role": "system", "content": ""}], "parent_ids": []}]
        [system] = self.workflow.insert(prompts)
        forced_tokens = torch.tensor([
            [101, 102, 103],
            [101, 102, 128009],
            [101, 128009, 0],
            [101, 128009, 0],
        ], dtype=torch.long)
        tasks: List[Task] = [{"header": ("assistant", None), "parent_ids": [system["id"]]} for _ in range(4)]
        out_tokens, out_nodes = get('tokens', 'nodes')(self.workflow.step(tasks, max_gen_len=4, teacher_force=forced_tokens))
        self.assertEqual(out_tokens, [[101, 102, 103], [101, 102], [101], [101]])
