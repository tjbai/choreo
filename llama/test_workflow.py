import torch
from unittest import TestCase
from llama.generation import Workflow, grouped_causal_mask

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)
        self.workflow.cur_id = 2
        self.workflow.id_map = torch.tensor([-1], dtype=torch.long)
        self.workflow.context = torch.tensor([128000])

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
