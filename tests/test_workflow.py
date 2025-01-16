import torch
from unittest import TestCase
from llama.workflow import Workflow, grouped_causal_mask, incremental_sequence_with_offset

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)

        class MockModel:
            def forward(self, *args, **kwargs):
                pass

        class MockFormatter:
            def encode_message(self, message):
                return [128006, 1, 128009]

            def encode_dialog_prompt(self, dialog, prefill):
                tokens = [128000]
                for _ in dialog:
                    tokens.extend([128006, 1, 128009])
                return tokens

        class MockTokenizer:
            bos_id = 128000

        self.workflow.model = MockModel()
        self.workflow.formatter = MockFormatter()
        self.workflow.tokenizer = MockTokenizer()
        self.workflow.max_seq_len = 128
        self.workflow.device = "cpu"
        self.workflow.max_nodes = 20
        self.workflow.max_parents = 20
        self.workflow.reset()

    def test_parent_mask(self):
        self.workflow.cache_len = 4
        self.workflow.context = torch.tensor([1, 2, 3, 4])
        self.workflow.node_map = torch.tensor([0, 1, 1, 2])
        self.workflow.cur_id = 3

        tasks = [
            {'parent_ids': [1], 'expects': ('assistant', None)},
            {'parent_ids': [2], 'expects': ('assistant', None)},
            {'parent_ids': [1, 2], 'expects': ('assistant', None)}
        ]

        self.workflow.add_nodes(tasks)
        self.assertTrue(torch.all(
            self.workflow.parent_map[3:6, :5] ==
            torch.tensor([
                [3, 1, 0, 0, 0],
                [4, 2, 0, 0, 0],
                [5, 1, 2, 0, 0]
            ])
        ))

        mask = self.workflow.dynamic_mask(self.workflow.parent_map[3:6])
        self.assertEqual(mask.shape, (3, 4))
        self.assertTrue(torch.all(
            mask == torch.tensor([
                [0, 0, 0, float("-inf")],
                [0, float("-inf"), float("-inf"), 0],
                [0, 0, 0, 0]
            ])
        ))

    def test_grouped_causal_mask(self):
        message_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
        mask = grouped_causal_mask(message_ids)
        self.assertEqual(mask.shape, (len(message_ids), len(message_ids)))
        self.assertTrue(torch.all(mask[:4, :4] == torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)))
        self.assertTrue(torch.all(mask[4:6, 4:6] == torch.triu(torch.full((2, 2), float("-inf")), diagonal=1)))
        self.assertTrue(torch.all(mask[6:, 6:] == torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)))

    def test_increment_sequence_with_offset(self):
        offsets = torch.tensor([10, 3, 7, 21])
        lengths = torch.tensor([3, 1, 4, 2])
        self.assertTrue(torch.all(incremental_sequence_with_offset(offsets, lengths) == torch.tensor([10, 11, 12, 3, 7, 8, 9, 10, 21, 22])))

    def test_insert(self):
        [system] = self.workflow.insert([
            {
                'message': {'role': 'system', 'content': ''},
                'parent_ids': []
            }
        ])

        self.assertEqual(system['id'], 1)
        self.assertEqual(self.workflow.cur_id, 2)
        self.assertTrue(torch.all(self.workflow.node_map[:4] == torch.tensor([0, 1, 1, 1])))
        self.assertTrue(torch.all(self.workflow.position_map[:4] == torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.all(self.workflow.context[:4] == torch.tensor([128000, 128006, 1, 128009])))
        self.assertTrue(torch.all(self.workflow.parent_map[1, :5] == torch.tensor([1, 0, 0, 0, 0])))

        user_1, user_2 = self.workflow.insert([
            {
                'message': {'role': 'user', 'content': ''},
                'parent_ids': [system['id']]
            },
            {
                'message': {'role': 'user', 'content': ''},
                'parent_ids': [system['id']]
            }
        ])

        self.assertEqual([user_1['id'], user_2['id']], [2, 3])
        self.assertEqual(self.workflow.cur_id, 4)
        self.assertTrue(torch.all(self.workflow.node_map[:10] == torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])))
        self.assertTrue(torch.all(self.workflow.position_map[:10] == torch.tensor([0, 1, 2, 3, 4, 5, 6, 4, 5, 6])))
        self.assertTrue(torch.all(self.workflow.parent_map[2:4, :3] == torch.tensor([[2, 1, 0], [3, 1, 0]])))

        _ = self.workflow.insert([
            {
                'message': {'role': 'user', 'content': ''},
                'parent_ids': [system['id'], user_1['id']]
            },
            {
                'message': {'role': 'user', 'content': ''},
                'parent_ids': [system['id'], user_1['id'], user_2['id']]
            }
        ])

        # this case is a bit tricky to define, expected behavior here might change
        self.assertTrue(torch.all(self.workflow.position_map[10:16] == torch.tensor([7, 8, 9, 10, 11, 12])))
        self.assertTrue(torch.all(self.workflow.parent_map[4:6, :5] == torch.tensor([[4, 1, 2, 0, 0], [5, 1, 2, 3, 0]])))
