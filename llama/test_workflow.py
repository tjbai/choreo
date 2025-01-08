import torch
from unittest import TestCase
from llama.generation import Workflow, grouped_causal_mask, incremental_sequence_with_offset

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)
        self.workflow.cur_id = 0
        self.workflow.id_map = torch.tensor([-1], dtype=torch.long)
        self.workflow.context = torch.tensor([128000])
        self.workflow.BOS_ID = 128000
        self.workflow.BOT_ID = 128006
        self.workflow.EOT_ID = 128009

        class MockModel:
            def forward(self, *args, **kwargs):
                pass

        class MockFormatter:
            def encode_dialog_prompt(self, dialog, prefill):
                tokens = [128000]
                for _ in dialog:
                    tokens.extend([128006, 1, 128009])
                return tokens

        self.workflow.model = MockModel()
        self.workflow.formatter = MockFormatter()

    def test_dependency_mask(self):
        self.workflow.context = torch.tensor([1, 2, 3, 4])
        self.workflow.id_map = torch.tensor([-1, 0, 0, 1])

        tasks = [
            {'requirements': [0], 'expects': ('assistant', None)},
            {'requirements': [1], 'expects': ('assistant', None)},
            {'requirements': [0, 1], 'expects': ('assistant', None)}
        ]

        mask = self.workflow._dependency_mask(tasks)
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
        dialog_1 = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}]
        dialog_2 = [{'role': 'system', 'content': ''}]

        ids = self.workflow.insert(dialog_1)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(self.workflow.cur_id, 2)
        self.assertTrue(torch.all(self.workflow.context == torch.tensor([128000, 128006, 1, 128009, 128006, 1, 128009])))
        self.assertTrue(torch.all(self.workflow.id_map == torch.tensor([-1, 0, 0, 0, 1, 1, 1])))

        ids = self.workflow.insert(dialog_2)
        self.assertEqual(ids, [2])
        self.assertEqual(self.workflow.cur_id, 3)
