from unittest import TestCase
import torch

from llama.workflow import Workflow

class TestWorkflowIntegration(TestCase):
    def setUp(self):
        class MockTransformer:
            def __init__(self):
                self.kv_cache = {}
                self.params = type('Params', (), {'max_seq_len': 2048})
                self.call_history = []

            def forward(self, tokens, start_pos, mask=None, position_ids=None):
                self.call_history.append({
                    'tokens': tokens.clone(),
                    'start_pos': start_pos,
                    'mask': mask.clone() if mask is not None else None,
                    'position_ids': position_ids.clone() if position_ids is not None else None
                })
                return torch.full((1, tokens.shape[1], 5000), 0.1)

            def reposition_cache(self, where, from_pos, to_pos):
                self.call_history.append({
                    'operation': 'reposition',
                    'where': where.clone(),
                    'from_pos': from_pos.clone(),
                    'to_pos': to_pos.clone()
                })

        class MockFormatter:
            def encode_message(self, message):
                return [2, 0, 2, 1, 0, 1]

            def encode_dialog(self, dialog):
                tokens = []
                for message in dialog:
                    tokens.extend(self.encode_message(message))
                return tokens

            def encode_header(self, message):
                return [2, 0, 2]

        class MockTokenizer:
            bos_id = 128000
            stop_tokens = [128001, 128009]

        self.model = MockTransformer()
        self.tokenizer = MockTokenizer()
        self.workflow = Workflow(
            self.model,
            self.tokenizer,
            max_nodes=10,
            max_parents=5
        )
        self.workflow.formatter = MockFormatter()

    def test_insert(self):
        prompts = [
            {
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': 'Hello'}
                ],
                'parent_ids': []
            }
        ]

        self.workflow.insert(prompts) # type: ignore
        self.assertEqual(self.model.call_history[-1]['start_pos'], 1)
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
            self.workflow.parent_map[1].tolist(),
            [1] + [0 for _ in range(4)]
        )

    def test_parallel_insert(self):
        system_prompts = [
            {
                'messages': [{'role': 'system', 'content': ''},],
                'parent_ids': []
            },
            {
                'messages': [{'role': 'system', 'content': ''},],
                'parent_ids': []
            }
        ]

        [sys1, sys2] = self.workflow.insert(system_prompts) # type: ignore
        self.assertEqual(self.model.call_history[-1]['start_pos'], 1)
        self.assertEqual(
            self.model.call_history[-1]['position_ids'].tolist(),
            2 * [1, 2, 3, 4, 5, 6]
        )

        user_prompts = [
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [sys1['id']]
            },
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [sys2['id']]
            },
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [sys1['id'], sys2['id']]
            }
        ]

        [user1, user2, user3] = self.workflow.insert(user_prompts) # type: ignore

        self.assertEqual(self.model.call_history[-1]['start_pos'], 13)
        self.assertEqual(
            self.model.call_history[-1]['position_ids'].tolist(),
            3 * [7, 8, 9, 10, 11, 12]
        )
        self.assertEqual(self.model.call_history[-1]['mask'].shape, (18, 18 + 13))
        self.assertEqual(
            self.workflow.position_map[13:31].tolist(),
            3 * [7, 8, 9, 10, 11, 12]
        )
        self.assertEqual(self.workflow.cache_len, 31)
