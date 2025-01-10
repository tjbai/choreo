import torch
from unittest import TestCase
from llama.workflow import Workflow, grouped_causal_mask, incremental_sequence_with_offset
from llama.model import precompute_freqs_cis, apply_rotary_emb, reposition_rotary_emb

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)
        self.workflow.cur_id = 0
        self.workflow.id_map = torch.tensor([-1], dtype=torch.long)
        self.workflow.context = torch.tensor([128000])
        self.workflow.device = "cpu"

        self.workflow.BOS_ID = 128000
        self.workflow.BOT_ID = 128006
        self.workflow.EOT_ID = 128009

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
        ids = self.workflow.insert([
            {
                'message': {'role': 'system', 'content': ''},
                'requirements': []
            }
        ])

        self.assertEqual(ids, [0])
        self.assertEqual(self.workflow.cur_id, 1)
        self.assertTrue(torch.all(self.workflow.context == torch.tensor([128000, 128006, 1, 128009])))
        self.assertTrue(torch.all(self.workflow.id_map == torch.tensor([-1, 0, 0, 0])))

        ids = self.workflow.insert([
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': ids
            },
            {
                'message': {'role': 'user', 'content': ''},
                'requirements': ids
            }
        ])

        self.assertEqual(ids, [1, 2])
        self.assertEqual(self.workflow.cur_id, 3)

    def setup_reposition(self):
        torch.manual_seed(42)
        n_layers, batch_size, seq_len, n_heads, head_dim = 16, 10, 4, 4, 8
        xq = torch.randn(n_layers, batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(n_layers, batch_size, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, 48)
        return xq, xk, freqs_cis

    def test_reposition_identity(self):
        xq, xk, freqs_cis = self.setup_reposition()
        xq_moved, xk_moved = reposition_rotary_emb(xq, xk, torch.ones(4).long(), torch.ones(4).long(), freqs_cis)
        self.assertEqual(xq_moved.shape, xq.shape)
        self.assertEqual(xk_moved.shape, xk.shape)
        self.assertTrue(torch.allclose(xq_moved, xq))
        self.assertTrue(torch.allclose(xk_moved, xk))

    def test_reposition_forwards_backwards(self):
        xq, xk, freqs_cis = self.setup_reposition()
        xq1, xk1 = reposition_rotary_emb(xq, xk, torch.zeros(4).long(), torch.ones(4).long(), freqs_cis)
        xq2, xk2 = reposition_rotary_emb(xq1, xk1, torch.ones(4).long(), torch.zeros(4).long(), freqs_cis)
        self.assertTrue(torch.allclose(xq2, xq))
        self.assertTrue(torch.allclose(xk2, xk))

    def test_reposition_cycle(self):
        xq, xk, freqs_cis = self.setup_reposition()
        pos0 = torch.zeros(4).long()
        pos1 = torch.ones(4).long()
        pos2 = torch.randint(10, (4,))

        xq1, xk1 = reposition_rotary_emb(xq, xk, pos0, pos1, freqs_cis)
        xq2, xk2 = reposition_rotary_emb(xq1, xk1, pos1, pos2, freqs_cis)
        xq1_back, xk1_back = reposition_rotary_emb(xq2, xk2, pos2, pos1, freqs_cis)
        xq0_back, xk0_back = reposition_rotary_emb(xq1_back, xk1_back, pos1, pos0, freqs_cis)
        self.assertTrue(torch.allclose(xq0_back, xq, atol=1e-5))
        self.assertTrue(torch.allclose(xk0_back, xk, atol=1e-5))

        from_pos = torch.tensor([0, 1, 2, 3])
        to_pos = torch.tensor([3, 1, 0, 2])
        xq_moved, xk_moved = reposition_rotary_emb(xq, xk, from_pos, to_pos, freqs_cis)
        xq_back, xk_back = reposition_rotary_emb(xq_moved, xk_moved, to_pos, from_pos, freqs_cis)
        self.assertTrue(torch.allclose(xq_back, xq, atol=1e-5))
        self.assertTrue(torch.allclose(xk_back, xk, atol=1e-5))

    def test_reposition_composition(self):
        xq, xk, freqs_cis = self.setup_reposition()
        start_pos = torch.tensor([0, 1, 2, 3])
        mid_pos = torch.tensor([1, 2, 3, 4])
        final_pos = torch.tensor([2, 3, 4, 5])
        xq1, xk2 = reposition_rotary_emb(xq, xk, start_pos, mid_pos, freqs_cis)
        xq2, xk2 = reposition_rotary_emb(xq1, xk2, mid_pos, final_pos, freqs_cis)
        xq_direct, xk_direct = reposition_rotary_emb(xq, xk, start_pos, final_pos, freqs_cis)
        self.assertTrue(torch.allclose(xq2, xq_direct, atol=1e-5))
        self.assertTrue(torch.allclose(xk2, xk_direct, atol=1e-5))

    def test_reposition_dtype(self):
        xq, xk, freqs_cis = self.setup_reposition()
        xq_half = xq.half()
        xk_half = xk.half()
        pos = torch.zeros(4).long()
        new_pos = torch.ones(4).long()
        xq_moved, xk_moved = reposition_rotary_emb(xq_half, xk_half, pos, new_pos, freqs_cis)
        self.assertEqual(xq_moved.dtype, torch.float16)
        self.assertEqual(xk_moved.dtype, torch.float16)
