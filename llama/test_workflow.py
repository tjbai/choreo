import torch
from unittest import TestCase
from llama.workflow import Workflow, grouped_causal_mask, incremental_sequence_with_offset
from llama.model import precompute_freqs_cis, apply_rotary_emb, reposition_rotary_emb

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)
        self.workflow.cur_id = 0
        self.workflow.id_map = torch.tensor([-1], dtype=torch.long)
        self.workflow.position_map = torch.tensor([0], dtype=torch.long)
        self.workflow.context = torch.tensor([128000])
        self.workflow.device = "cpu"

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
        [system] = self.workflow.insert([
            {
                'message': {'role': 'system', 'content': ''},
                'requirements': []
            }
        ])

        self.assertEqual(system, 0)
        self.assertEqual(self.workflow.cur_id, 1)
        self.assertTrue(torch.all(self.workflow.id_map == torch.tensor([-1, 0, 0, 0])))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.all(self.workflow.context == torch.tensor([128000, 128006, 1, 128009])))

        user_1, user_2 = self.workflow.insert([
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': [system]
            },
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': [system]
            }
        ])

        self.assertEqual([user_1, user_2], [1, 2])
        self.assertEqual(self.workflow.cur_id, 3)
        self.assertTrue(torch.all(self.workflow.id_map == torch.tensor([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2])))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 1, 2, 3, 4, 5, 6, 4, 5, 6])))

        _ = self.workflow.insert([
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': [system, user_1]
            },
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': [system, user_1, user_2]
            }
        ])

        # this case is a bit tricky to define, expected behavior here might change
        self.assertTrue(torch.all(self.workflow.position_map[-6:] == torch.tensor([7, 8, 9, 10, 11, 12])))

    def setup_reposition(self):
        torch.manual_seed(42)
        n_layers, batch_size, seq_len, n_heads, head_dim = 16, 10, 4, 4, 8
        xk = torch.randn(n_layers, batch_size, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, 48)
        return xk, freqs_cis

    def test_reposition_identity(self):
        xk, freqs_cis = self.setup_reposition()
        xk_moved = reposition_rotary_emb(xk, torch.ones(4).long(), torch.ones(4).long(), freqs_cis)
        self.assertEqual(xk_moved.shape, xk.shape)
        self.assertTrue(torch.allclose(xk_moved, xk))

    def test_reposition_forwards_backwards(self):
        xk, freqs_cis = self.setup_reposition()
        xk1 = reposition_rotary_emb(xk, torch.zeros(4).long(), torch.ones(4).long(), freqs_cis)
        xk2 = reposition_rotary_emb(xk1, torch.ones(4).long(), torch.zeros(4).long(), freqs_cis)
        self.assertTrue(torch.allclose(xk2, xk))

    def test_reposition_cycle(self):
        xk, freqs_cis = self.setup_reposition()
        pos0 = torch.zeros(4).long()
        pos1 = torch.ones(4).long()
        pos2 = torch.randint(10, (4,))

        xk1 = reposition_rotary_emb(xk, pos0, pos1, freqs_cis)
        xk2 = reposition_rotary_emb(xk1, pos1, pos2, freqs_cis)
        xk1_back = reposition_rotary_emb(xk2, pos2, pos1, freqs_cis)
        xk0_back = reposition_rotary_emb(xk1_back, pos1, pos0, freqs_cis)
        self.assertTrue(torch.allclose(xk0_back, xk, atol=1e-5))

        from_pos = torch.tensor([0, 1, 2, 3])
        to_pos = torch.tensor([3, 1, 0, 2])
        xk_moved = reposition_rotary_emb(xk, from_pos, to_pos, freqs_cis)
        xk_back = reposition_rotary_emb(xk_moved, to_pos, from_pos, freqs_cis)
        self.assertTrue(torch.allclose(xk_back, xk, atol=1e-5))

    def test_reposition_composition(self):
        xk, freqs_cis = self.setup_reposition()
        start_pos = torch.tensor([0, 1, 2, 3])
        mid_pos = torch.tensor([1, 2, 3, 4])
        final_pos = torch.tensor([2, 3, 4, 5])
        xk2 = reposition_rotary_emb(xk, start_pos, mid_pos, freqs_cis)
        xk2 = reposition_rotary_emb(xk2, mid_pos, final_pos, freqs_cis)
        xk_direct = reposition_rotary_emb(xk, start_pos, final_pos, freqs_cis)
        self.assertTrue(torch.allclose(xk2, xk_direct, atol=1e-5))

    def test_reposition_equality(self):
        xk, freqs_cis = self.setup_reposition()

        start_pos = torch.zeros(4).long()
        mid_pos = torch.tensor([1, 2, 3, 4])
        final_pos = torch.tensor([2, 3, 4, 5])

        xk1 = reposition_rotary_emb(xk, start_pos, mid_pos, freqs_cis)
        _, xk1_apply = apply_rotary_emb(torch.zeros_like(xk), xk, freqs_cis[mid_pos])
        self.assertTrue(torch.allclose(xk1, xk1_apply, atol=1e-5))

        xk2 = reposition_rotary_emb(xk1, mid_pos, final_pos, freqs_cis)
        _, xk2_apply = apply_rotary_emb(torch.zeros_like(xk), xk, freqs_cis[final_pos])
        self.assertTrue(torch.allclose(xk2, xk2_apply, atol=1e-5))
